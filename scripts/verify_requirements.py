"""One-click visual & JSON evaluator verification runner for Mission A3-1 requirements."""

from __future__ import annotations

import ast
import json
import re
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import ScanConfig
from src.eda import run_dataset_eda
from src.evaluator import evaluate_dataset, run_parameter_sweep
from src.geometry import order_points, order_points_naive_sum_diff
from src.io_utils import configure_utf8_console, save_json
from src.pipeline import DocumentScanner

FORBIDDEN_LIBRARIES = {"torch", "tensorflow", "keras", "pytesseract", "easyocr", "scipy", "shapely"}


def _check_forbidden_imports() -> Dict[str, Any]:
    violations: List[str] = []
    src_files = sorted((PROJECT_ROOT / "src").glob("*.py"))
    for py_file in src_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN_LIBRARIES:
                        violations.append(f"{py_file.name}:{root}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_LIBRARIES:
                    violations.append(f"{py_file.name}:{root}")
    return {"passed": len(violations) == 0, "checked_files": len(src_files), "violations": violations}


def _check_relative_markdown_images() -> Dict[str, Any]:
    md_files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "스터디.md",
        PROJECT_ROOT / "REPORT.md",
    ]
    abs_violations: List[str] = []
    missing_targets: List[str] = []
    found_images = 0

    pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for md in md_files:
        if not md.exists():
            continue
        content = md.read_text(encoding="utf-8")
        for match in pattern.finditer(content):
            link = match.group(1).strip()
            found_images += 1
            if link.startswith(("file://", "http://", "https://", "/")) or re.match(r"^[A-Za-z]:\\", link):
                abs_violations.append(f"{md.name} -> {link}")
            else:
                resolved = (md.parent / link).resolve()
                if not resolved.exists():
                    missing_targets.append(f"{md.name} -> {link}")
    return {
        "passed": len(abs_violations) == 0 and len(missing_targets) == 0 and found_images > 0,
        "embedded_images": found_images,
        "absolute_path_violations": abs_violations,
        "missing_targets": missing_targets,
    }


def _run_unit_tests() -> Dict[str, Any]:
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    elapsed = round(time.perf_counter() - start, 4)
    return {
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "elapsed_sec": elapsed,
    }


def _check_manual_geometry_tie_break() -> Dict[str, Any]:
    diamond = np.array([[100, 0], [200, 100], [100, 200], [0, 100]], dtype=np.float32)
    naive = order_points_naive_sum_diff(diamond)
    ours = order_points(diamond)
    naive_unique = int(len(np.unique(naive, axis=0)))
    ours_unique = int(len(np.unique(ours, axis=0)))
    return {
        "passed": ours_unique == 4 and naive_unique < 4,
        "naive_unique_vertices": naive_unique,
        "ours_unique_vertices": ours_unique,
    }


def run_verification() -> Dict[str, Any]:
    """Execute all requirement checks and print a visual summary table + JSON report."""
    configure_utf8_console()
    meta_path = PROJECT_ROOT / "data" / "flow_document_images" / "metadata_flow.json"

    checks: List[Dict[str, Any]] = []

    c1 = _check_forbidden_imports()
    checks.append({"id": "REQ-01", "name": "금지 라이브러리 미사용 (Zero DL/OCR/Scipy)", **c1})

    c2 = _check_manual_geometry_tie_break()
    checks.append({"id": "REQ-02", "name": "수동 4점 좌표 정렬 및 45도 동률 방어", **c2})

    c3 = _run_unit_tests()
    checks.append({"id": "REQ-03", "name": "자동화 Mock 단위 테스트 100% 통과", **c3})

    eda = run_dataset_eda(meta_path)
    checks.append(
        {
            "id": "REQ-04",
            "name": "20장 테스트셋 밝기 히스토그램 EDA",
            "passed": eda["total_images"] == 20,
            "total_images": eda["total_images"],
            "easy_std": eda["difficulty_summary"]["easy"]["std_brightness"],
            "hard_std": eda["difficulty_summary"]["hard"]["std_brightness"],
        }
    )

    eval_res = evaluate_dataset(meta_path, config=ScanConfig())
    checks.append(
        {
            "id": "REQ-05",
            "name": "20장 정량적 성능 평가 (4조건 × 5장)",
            "passed": eval_res["overall"]["total"] == 20 and eval_res["by_condition"][0]["success"] == 5,
            "overall": eval_res["overall"],
            "by_condition": eval_res["by_condition"],
        }
    )

    sweep = run_parameter_sweep(meta_path)
    checks.append(
        {
            "id": "REQ-06",
            "name": "다중 프리셋 파라미터 딜레마 실증",
            "passed": bool(sweep["parameter_dilemma_proven"]),
            "best_single_preset": sweep["best_single_preset_success"],
            "oracle_union": sweep["oracle_union_success"],
            "conflict_pairs": len(sweep["conflict_pairs"]),
        }
    )

    # Single scan end-to-end check on easy_01 (Google Flow photorealistic set)
    sample_res = DocumentScanner().scan_file(PROJECT_ROOT / "data" / "flow_document_images" / "easy_01.jpg")
    checks.append(
        {
            "id": "REQ-07",
            "name": "단일 이미지 E2E 스캔 + 적응형 이진화",
            "passed": bool(sample_res.detected and sample_res.scanned is not None),
            "area_ratio": round(sample_res.area_ratio, 4),
            "scanned_shape": list(sample_res.scanned.shape) if sample_res.scanned is not None else None,
        }
    )

    c8 = _check_relative_markdown_images()
    checks.append({"id": "REQ-08", "name": "Matplotlib 다이어그램 상대경로 임베딩 검증", **c8})

    required_docs = [
        "PLAN.md",
        "스터디.md",
        "REPORT.md",
        "README.md",
        "NOTEBOOKLM_GUIDE.md",
        "NOTEBOOKLM_FULL_SOURCE.txt",
        "study/핵심_코드_스터디.md",
    ]
    missing_docs = [d for d in required_docs if not (PROJECT_ROOT / d).exists()]
    checks.append(
        {
            "id": "REQ-09",
            "name": "8단계 마스터 워크플로우 필수 산출물 존재",
            "passed": len(missing_docs) == 0,
            "missing_docs": missing_docs,
        }
    )

    all_passed = all(c["passed"] for c in checks)

    print("=" * 84)
    print("  CODYSSEY A3-01 | OPENCV DOCUMENT SCANNER - AUTOMATED REQUIREMENT VERIFIER")
    print("=" * 84)
    print(f"{'ID':<8} | {'STATUS':<8} | {'REQUIREMENT CHECK'}")
    print("-" * 84)
    for item in checks:
        status = "[PASS]" if item["passed"] else "[FAIL]"
        print(f"{item['id']:<8} | {status:<8} | {item['name']}")
    print("-" * 84)
    print("  [20-IMAGE BENCHMARK SUMMARY (Default Preset: blur=5, Canny=50/150)]")
    print(f"  {'조건':<22} | {'테스트':>6} | {'성공':>4} | {'실패':>4} | {'성공률':>7}")
    print("  " + "-" * 58)
    for row in eval_res["by_condition"]:
        print(
            f"  {row['condition']:<20} | {row['total']:>6} | {row['success']:>4} | "
            f"{row['fail']:>4} | {row['success_rate']:>6.1f}%"
        )
    ov = eval_res["overall"]
    print("  " + "-" * 58)
    print(f"  {'전체 합계':<20} | {ov['total']:>6} | {ov['success']:>4} | {ov['fail']:>4} | {ov['success_rate']:>6.1f}%")
    print("=" * 84)
    print(f"  OVERALL VERIFICATION: {'ALL PASS (100%)' if all_passed else 'NEEDS ATTENTION'}")
    print("=" * 84)

    report = {
        "all_passed": all_passed,
        "checks": checks,
    }
    save_json(PROJECT_ROOT / "outputs" / "verification_report.json", report)
    print("\n[JSON SUMMARY]")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    res = run_verification()
    sys.exit(0 if res["all_passed"] else 1)
