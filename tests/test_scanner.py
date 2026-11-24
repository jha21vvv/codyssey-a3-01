"""Fast, zero-network automated unit tests for the OpenCV Document Scanner."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src.cli import main as cli_main
from src.config import PRESETS, ScanConfig, normalize_odd_kernel
from src.contour_detector import find_document_contour
from src.eda import analyze_image_brightness, compute_histogram
from src.geometry import (
    compute_corner_error,
    compute_output_dimensions,
    compute_quad_iou,
    four_point_transform,
    order_points,
    order_points_naive_sum_diff,
)
from src.gui import run_interactive_gui, run_webcam_mode
from src.io_utils import load_image, load_json, save_image, save_json
from src.pipeline import DocumentScanner, build_stage_montage, save_stage_outputs
from src.postprocessor import apply_adaptive_threshold
from src.preprocessor import apply_gaussian_blur, detect_canny_edges, preprocess_image, to_grayscale


def _make_clean_synthetic_doc(
    width: int = 500,
    height: int = 400,
    corners: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an in-memory crisp synthetic document on a dark background for fast testing."""
    canvas = np.full((height, width, 3), 35, dtype=np.uint8)
    pts = np.array(
        corners or [[120, 50], [380, 60], [390, 340], [110, 330]],
        dtype=np.float32,
    )
    cv2.fillConvexPoly(canvas, pts.astype(np.int32), (242, 242, 245))
    cv2.putText(canvas, "TEST DOC", (165, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return canvas, pts


class TestConfigAndIO(unittest.TestCase):
    """Test parameter normalization and Unicode/Korean path safe I/O."""

    def test_odd_kernel_normalization(self) -> None:
        self.assertEqual(normalize_odd_kernel(4), 5)
        self.assertEqual(normalize_odd_kernel(0), 1)
        self.assertEqual(normalize_odd_kernel(52, maximum=51), 51)
        self.assertEqual(normalize_odd_kernel(7), 7)

    def test_config_auto_fixes_inverted_canny_and_even_kernels(self) -> None:
        cfg = ScanConfig(blur_kernel=6, canny_low=200, canny_high=50, morph_kernel=4, adaptive_block_size=14)
        self.assertEqual(cfg.blur_kernel, 7)
        self.assertEqual(cfg.canny_low, 50)
        self.assertEqual(cfg.canny_high, 200)
        self.assertEqual(cfg.morph_kernel, 5)
        self.assertEqual(cfg.adaptive_block_size, 15)

    def test_config_equal_canny_thresholds_get_minimum_gap(self) -> None:
        cfg = ScanConfig(canny_low=100, canny_high=100)
        self.assertEqual(cfg.canny_low, 100)
        self.assertEqual(cfg.canny_high, 101)

    def test_unicode_korean_path_save_and_load(self) -> None:
        img, _ = _make_clean_synthetic_doc(160, 120, [[20, 20], [140, 20], [140, 100], [20, 100]])
        with tempfile.TemporaryDirectory() as tmpdir:
            korean_path = Path(tmpdir) / "한글 폴더" / "영수증_테스트 이미지.png"
            saved = save_image(korean_path, img)
            self.assertTrue(saved.exists())
            reloaded = load_image(saved)
            self.assertEqual(reloaded.shape, img.shape)
            self.assertTrue(np.array_equal(reloaded, img))

    def test_missing_or_corrupt_image_raises_clean_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "does_not_exist.png"
            with self.assertRaises(FileNotFoundError):
                load_image(missing)

            empty_file = Path(tmpdir) / "empty.png"
            empty_file.write_bytes(b"")
            with self.assertRaises(ValueError):
                load_image(empty_file)

            corrupt_file = Path(tmpdir) / "corrupt.png"
            corrupt_file.write_bytes(b"not-a-valid-image-payload")
            with self.assertRaises(ValueError):
                load_image(corrupt_file)

    def test_json_roundtrip_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jp = Path(tmpdir) / "메타데이터.json"
            payload = {"조건": "단순 배경 + 정면", "성공률": 100.0}
            save_json(jp, payload)
            self.assertEqual(load_json(jp), payload)


class TestGeometryManualOrdering(unittest.TestCase):
    """Test pure geometric 4-point ordering, diamond tie-breaking, and perspective warp."""

    def test_order_points_scrambled_rectangle(self) -> None:
        pts = np.array([[300, 400], [100, 50], [100, 400], [300, 50]], dtype=np.float32)
        ordered = order_points(pts)
        expected = np.array([[100, 50], [300, 50], [300, 400], [100, 400]], dtype=np.float32)
        np.testing.assert_allclose(ordered, expected, atol=1e-4)

    def test_order_points_45_degree_diamond_beats_naive_sum_diff(self) -> None:
        # Exact 45-degree diamond where x+y and y-x have ties!
        diamond = np.array([[100, 0], [200, 100], [100, 200], [0, 100]], dtype=np.float32)
        naive = order_points_naive_sum_diff(diamond)
        # Naive method duplicates vertices on 45-degree symmetry
        unique_naive = np.unique(naive, axis=0)
        self.assertLess(len(unique_naive), 4)

        # Our geometric polar-angle algorithm preserves all 4 unique points in CW order
        ordered = order_points(diamond)
        unique_ordered = np.unique(ordered, axis=0)
        self.assertEqual(len(unique_ordered), 4)
        expected = np.array([[100, 0], [200, 100], [100, 200], [0, 100]], dtype=np.float32)
        np.testing.assert_allclose(ordered, expected, atol=1e-4)

    def test_order_points_rejects_collinear_and_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            order_points(np.array([[0, 0], [10, 10], [20, 20], [30, 30]], dtype=np.float32))
        with self.assertRaises(ValueError):
            order_points(np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32))
        with self.assertRaises(ValueError):
            order_points(np.array([[0, 0], [np.nan, 0], [10, 10], [0, 10]], dtype=np.float32))

    def test_four_point_transform_dimensions_and_iou(self) -> None:
        img, pts = _make_clean_synthetic_doc()
        warped, ordered, matrix = four_point_transform(img, pts)
        w, h = compute_output_dimensions(ordered)
        self.assertEqual(warped.shape[:2], (h, w))
        self.assertEqual(matrix.shape, (3, 3))
        self.assertAlmostEqual(compute_quad_iou(ordered, pts, img.shape[:2]), 1.0, places=3)
        self.assertAlmostEqual(compute_corner_error(ordered, pts), 0.0, places=3)


class TestPipelinePreprocessAndPostprocess(unittest.TestCase):
    """Test preprocessing, contour detection, adaptive threshold, and full pipeline."""

    def test_grayscale_handles_bgr_bgra_and_2d(self) -> None:
        bgr = np.zeros((20, 30, 3), dtype=np.uint8)
        bgra = np.zeros((20, 30, 4), dtype=np.uint8)
        gray = np.zeros((20, 30), dtype=np.uint8)
        self.assertEqual(to_grayscale(bgr).shape, (20, 30))
        self.assertEqual(to_grayscale(bgra).shape, (20, 30))
        self.assertEqual(to_grayscale(gray).shape, (20, 30))

    def test_full_pipeline_detects_clean_document(self) -> None:
        img, gt_pts = _make_clean_synthetic_doc()
        scanner = DocumentScanner()
        res = scanner.scan_image(img)
        self.assertTrue(res.detected)
        self.assertIsNotNone(res.ordered_quad)
        self.assertIsNotNone(res.warped)
        self.assertIsNotNone(res.scanned)
        self.assertGreater(compute_quad_iou(res.ordered_quad, gt_pts, img.shape[:2]), 0.92)
        self.assertLess(compute_corner_error(res.ordered_quad, gt_pts), 6.0)
        # Adaptive threshold output must be binary (only 0 and 255)
        unique_vals = set(np.unique(res.scanned).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_full_pipeline_graceful_failure_on_blank_image(self) -> None:
        blank = np.full((300, 400, 3), 120, dtype=np.uint8)
        scanner = DocumentScanner()
        res = scanner.scan_image(blank)
        self.assertFalse(res.detected)
        self.assertIsNone(res.scanned)
        self.assertEqual(res.failure_reason, "no_contours_found")

    def test_montage_and_stage_output_saving(self) -> None:
        img, _ = _make_clean_synthetic_doc(320, 240, [[60, 30], [260, 35], [265, 210], [55, 205]])
        res = DocumentScanner().scan_image(img)
        montage = build_stage_montage(res, tile_size=(160, 120))
        self.assertEqual(montage.shape[1], 160 * 3)
        self.assertEqual(montage.shape[0], (120 + 32) * 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_stage_outputs(res, tmpdir, stem="unit_sample")
            self.assertIn("montage", files)
            self.assertIn("scanned", files)
            for p in files.values():
                self.assertTrue(p.exists())

    def test_eda_histogram_and_stats(self) -> None:
        img, _ = _make_clean_synthetic_doc(200, 160, [[40, 20], [160, 20], [160, 140], [40, 140]])
        stats, hist = analyze_image_brightness(img, "t1", "t1.jpg", "단순 배경 + 정면", "easy")
        self.assertEqual(hist.shape, (256,))
        self.assertAlmostEqual(float(hist.sum()), 1.0, places=5)
        self.assertGreater(stats.dynamic_range, 100.0)


class TestMockedGuiWebcamAndCli(unittest.TestCase):
    """Zero-GUI / Zero-Camera mocked tests verifying trackbar GUI, webcam mode, and CLI."""

    def test_interactive_gui_trackbar_and_save_mocked(self) -> None:
        img, _ = _make_clean_synthetic_doc(260, 200, [[50, 30], [210, 30], [210, 170], [50, 170]])
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = save_image(Path(tmpdir) / "doc.png", img)
            out_path = Path(tmpdir) / "saved_from_gui.png"

            with (
                mock.patch("src.gui.cv2.namedWindow"),
                mock.patch("src.gui.cv2.resizeWindow"),
                mock.patch("src.gui.cv2.createTrackbar") as mock_trackbar,
                mock.patch("src.gui.cv2.getTrackbarPos", side_effect=[5, 50, 150, 0, 7, 40, 140, 3]),
                mock.patch("src.gui.cv2.imshow") as mock_imshow,
                mock.patch("src.gui.cv2.waitKey", side_effect=[ord("s"), ord("q")]),
                mock.patch("src.gui.cv2.destroyAllWindows") as mock_destroy,
            ):
                result = run_interactive_gui(in_path, output_path=out_path, max_iterations=5)
                self.assertTrue(result.detected)
                self.assertGreaterEqual(mock_trackbar.call_count, 3)
                self.assertGreaterEqual(mock_imshow.call_count, 5)
                mock_destroy.assert_called_once()
                self.assertTrue(out_path.exists())

    def test_webcam_mode_capture_s_and_quit_q_mocked(self) -> None:
        frame, _ = _make_clean_synthetic_doc(260, 200, [[50, 30], [210, 30], [210, 170], [50, 170]])
        fake_cap = mock.MagicMock()
        fake_cap.isOpened.return_value = True
        fake_cap.read.side_effect = [(True, frame), (True, frame)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch("src.gui.cv2.VideoCapture", return_value=fake_cap),
                mock.patch("src.gui.cv2.imshow"),
                mock.patch("src.gui.cv2.waitKey", side_effect=[ord("s"), ord("q")]),
                mock.patch("src.gui.cv2.destroyAllWindows") as mock_destroy,
            ):
                saved = run_webcam_mode(camera_index=0, output_dir=tmpdir, max_frames=5)
                self.assertEqual(len(saved), 1)
                self.assertTrue(saved[0].exists())
                fake_cap.release.assert_called_once()
                mock_destroy.assert_called_once()

    def test_webcam_unavailable_raises_friendly_error(self) -> None:
        fake_cap = mock.MagicMock()
        fake_cap.isOpened.return_value = False
        with mock.patch("src.gui.cv2.VideoCapture", return_value=fake_cap):
            with self.assertRaises(RuntimeError):
                run_webcam_mode(camera_index=9)

    def test_cli_direct_image_path_shorthand(self) -> None:
        img, _ = _make_clean_synthetic_doc(260, 200, [[50, 30], [210, 30], [210, 170], [50, 170]])
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = save_image(Path(tmpdir) / "direct.png", img)
            out_path = Path(tmpdir) / "out.png"
            code = cli_main([str(in_path), "-o", str(out_path)])
            self.assertEqual(code, 0)
            self.assertTrue(out_path.exists())


class TestConstraintCompliance(unittest.TestCase):
    """Static AST check verifying zero forbidden libraries are imported anywhere in src/."""

    FORBIDDEN_ROOTS = {
        "torch",
        "tensorflow",
        "keras",
        "pytesseract",
        "easyocr",
        "scipy",
        "shapely",
    }

    def test_no_forbidden_imports_in_src(self) -> None:
        src_dir = Path(__file__).resolve().parent.parent / "src"
        py_files = sorted(src_dir.glob("*.py"))
        self.assertGreaterEqual(len(py_files), 8)

        for py_file in py_files:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        self.assertNotIn(root, self.FORBIDDEN_ROOTS, f"Forbidden import {root} in {py_file.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root = node.module.split(".")[0]
                    self.assertNotIn(root, self.FORBIDDEN_ROOTS, f"Forbidden import {root} in {py_file.name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
