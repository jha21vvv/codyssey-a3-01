"""Generate a deterministic 20-image realistic document benchmark dataset (4 conditions x 5)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import configure_utf8_console, save_image, save_json


def _create_document_texture(
    doc_type: str,
    width: int = 420,
    height: int = 560,
    paper_color: Tuple[int, int, int] = (244, 245, 247),
) -> np.ndarray:
    """Render a crisp synthetic document (A4 report, receipt, invoice, medical form, card)."""
    doc = np.full((height, width, 3), paper_color, dtype=np.uint8)

    if doc_type == "receipt":
        cv2.putText(doc, "CODYSSEY MART", (85, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.line(doc, (30, 82), (width - 30, 82), (60, 60, 60), 2)
        items = [
            ("USB-C Cable", "12,900"),
            ("Notebook A5", "6,500"),
            ("Coffee Beans", "18,000"),
            ("Desk Lamp", "34,500"),
            ("Thermal Paper", "4,200"),
            ("Subtotal", "76,100"),
            ("VAT (10%)", "7,610"),
            ("TOTAL", "83,710"),
        ]
        y = 125
        for idx, (name, price) in enumerate(items):
            thickness = 2 if idx == len(items) - 1 else 1
            cv2.putText(doc, name, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), thickness, cv2.LINE_AA)
            cv2.putText(doc, price, (width - 135, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), thickness, cv2.LINE_AA)
            y += 42
            if idx == 4:
                cv2.line(doc, (30, y - 20), (width - 30, y - 20), (90, 90, 90), 1)
        for bx in range(55, width - 55, 7):
            w_bar = 2 if (bx // 7) % 3 == 0 else 4
            cv2.rectangle(doc, (bx, height - 90), (bx + w_bar, height - 35), (30, 30, 30), -1)
    elif doc_type == "business_card":
        cv2.rectangle(doc, (0, 0), (width, 70), (45, 75, 130), -1)
        cv2.putText(doc, "ANTIGRAVITY LABS", (30, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 248, 252), 2, cv2.LINE_AA)
        cv2.putText(doc, "JAEHYUN AN", (32, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (25, 25, 30), 2, cv2.LINE_AA)
        cv2.putText(doc, "Computer Vision Engineer", (32, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 75, 85), 1, cv2.LINE_AA)
        cv2.line(doc, (32, 200), (width - 32, 200), (140, 150, 165), 1)
        cv2.putText(doc, "Email: vision@codyssey.ai", (32, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(doc, "Tel: +82-10-2026-0301", (32, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (40, 40, 40), 1, cv2.LINE_AA)
    else:
        header_map = {
            "a4_report": "INSPECTION REPORT #A3-01",
            "medical_form": "MEDICAL PRESCRIPTION",
            "invoice": "COMMERCIAL INVOICE",
        }
        title = header_map.get(doc_type, "DOCUMENT SCAN SAMPLE")
        cv2.rectangle(doc, (28, 28), (width - 28, 84), (225, 232, 242), -1)
        cv2.putText(doc, title, (40, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (25, 35, 55), 2, cv2.LINE_AA)
        cv2.line(doc, (28, 96), (width - 28, 96), (55, 65, 80), 2)

        for row in range(10):
            y = 132 + row * 34
            cv2.putText(
                doc,
                f"Section {row + 1:02d}: Verified pipeline parameter check OK",
                (34, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (35, 38, 42),
                1,
                cv2.LINE_AA,
            )
        cv2.rectangle(doc, (34, height - 105), (width - 34, height - 36), (90, 95, 105), 1)
        cv2.line(doc, (width // 2, height - 105), (width // 2, height - 36), (90, 95, 105), 1)
        cv2.putText(doc, "DATE: 2026-11-24", (46, height - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1, cv2.LINE_AA)
        cv2.putText(doc, "SIGN: APPROVED", (width // 2 + 16, height - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1, cv2.LINE_AA)

    return doc


def _make_background(
    style: str,
    rng: np.random.Generator,
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """Synthesize realistic desk/table backgrounds of varying visual complexity."""
    if style == "dark_matte":
        base = np.full((height, width, 3), (42, 46, 52), dtype=np.float32)
        noise = rng.normal(0, 3.5, size=(height, width, 3))
        return np.clip(base + noise, 0, 255).astype(np.uint8)

    if style == "dark_wood":
        y_coords = np.linspace(0, 14 * np.pi, height)[:, None]
        grain = (np.sin(y_coords) * 8 + np.sin(y_coords * 2.7 + 0.8) * 5)[:, :, None]
        base = np.full((height, width, 3), (38, 52, 74), dtype=np.float32) + grain
        noise = rng.normal(0, 4.0, size=(height, width, 3))
        return np.clip(base + noise, 0, 255).astype(np.uint8)

    if style == "wood_clutter":
        bg = np.full((height, width, 3), (62, 84, 110), dtype=np.float32)
        for y in range(0, height, 6):
            delta = float(rng.uniform(-12, 12))
            bg[y : y + 3, :] += delta
        bg = np.clip(bg + rng.normal(0, 6, size=bg.shape), 0, 255).astype(np.uint8)
        cv2.rectangle(bg, (18, 25), (150, 560), (45, 50, 65), -1)
        cv2.rectangle(bg, (660, 40), (775, 220), (80, 125, 165), -1)
        return bg

    if style == "grid_plaid":
        bg = np.full((height, width, 3), (95, 102, 112), dtype=np.uint8)
        for x in range(0, width, 24):
            cv2.line(bg, (x, 0), (x, height), (45, 52, 62), 2)
        for y in range(0, height, 24):
            cv2.line(bg, (0, y), (width, y), (45, 52, 62), 2)
        for x in range(12, width, 48):
            cv2.line(bg, (x, 0), (x, height), (145, 150, 158), 1)
        cv2.rectangle(bg, (30, 40), (140, 540), (65, 70, 82), -1)
        cv2.line(bg, (680, 60), (745, 520), (35, 45, 95), 14)
        return bg

    if style == "beige_low_contrast":
        base = np.full((height, width, 3), (212, 218, 224), dtype=np.float32)
        noise = rng.normal(0, 6.5, size=(height, width, 3))
        bg = np.clip(base + noise, 0, 255).astype(np.uint8)
        for x in range(0, width, 60):
            cv2.line(bg, (x, 0), (x, height), (196, 202, 208), 1)
        return bg

    if style == "cluttered_desk":
        bg = np.full((height, width, 3), (88, 96, 108), dtype=np.uint8)
        for _ in range(38):
            x1, y1 = int(rng.integers(0, width - 80)), int(rng.integers(0, height - 80))
            x2, y2 = x1 + int(rng.integers(30, 160)), y1 + int(rng.integers(30, 140))
            col = tuple(int(c) for c in rng.integers(40, 215, size=3))
            cv2.rectangle(bg, (x1, y1), (x2, y2), col, 2)
        for _ in range(25):
            p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            col = tuple(int(c) for c in rng.integers(30, 220, size=3))
            cv2.line(bg, p1, p2, col, int(rng.integers(1, 4)))
        return bg

    return np.full((height, width, 3), (50, 50, 55), dtype=np.uint8)


def _composite_document(
    background: np.ndarray,
    document: np.ndarray,
    dst_corners: np.ndarray,
) -> np.ndarray:
    """Perspective-warp document texture onto background canvas cleanly."""
    dh, dw = document.shape[:2]
    src = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], dtype=np.float32)
    dst = np.asarray(dst_corners, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)

    h, w = background.shape[:2]
    warped_doc = cv2.warpPerspective(document, matrix, (w, h), flags=cv2.INTER_LINEAR)
    mask = np.full((dh, dw), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)

    out = background.copy()
    out[warped_mask > 127] = warped_doc[warped_mask > 127]
    return out


def _apply_diagonal_shadow(
    image: np.ndarray,
    strength: float,
    blur_ksize: int,
    slope: float = 0.65,
    offset: float = 120.0,
) -> np.ndarray:
    """Multiply image luminance by a smooth or hard diagonal cast-shadow mask."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shadow_region = (yy > slope * xx + offset).astype(np.float32)
    if blur_ksize >= 3:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        shadow_region = cv2.GaussianBlur(shadow_region, (k, k), 0)
    factor = 1.0 - strength * shadow_region
    result = image.astype(np.float32) * factor[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_spotlight_glare(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: float,
    intensity: float,
) -> np.ndarray:
    """Add specular reflection glare washing out local edges."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    glare = intensity * np.exp(-dist_sq / (2.0 * (radius**2)))
    out = image.astype(np.float32) + glare[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_crease_folds(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Simulate physical paper crumple/fold shading lines across document."""
    out = image.copy().astype(np.float32)
    h, w = out.shape[:2]
    cx = float(np.mean(corners[:, 0]))
    cy = float(np.mean(corners[:, 1]))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    band1 = np.exp(-((yy - (0.35 * (xx - cx) + cy)) ** 2) / 140.0)
    band2 = np.exp(-((yy - (-0.45 * (xx - cx) + cy + 35)) ** 2) / 110.0)
    out -= (band1 * 55.0 + band2 * 48.0)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_dataset(output_root: Path | None = None) -> Dict[str, Any]:
    """Build 20 benchmark test images across 4 conditions and save metadata.json."""
    root = output_root or (PROJECT_ROOT / "data")
    img_dir = root / "test_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260301)

    specs: List[Dict[str, Any]] = [
        # 1. 단순 배경 + 정면 (5장, easy)
        {
            "id": "easy_01",
            "condition": "단순 배경 + 정면",
            "difficulty": "easy",
            "doc_type": "a4_report",
            "doc_size": (400, 520),
            "bg": "dark_matte",
            "corners": [[200, 55], [600, 58], [602, 548], [198, 545]],
            "lighting": "균일 조명 (실내 LED)",
            "angle_deg": 2,
            "background_desc": "단색 어두운 매트",
            "notes": "표준 정면 A4 검사 보고서",
        },
        {
            "id": "easy_02",
            "condition": "단순 배경 + 정면",
            "difficulty": "easy",
            "doc_type": "receipt",
            "doc_size": (320, 520),
            "bg": "dark_wood",
            "corners": [[245, 50], [555, 56], [560, 542], [240, 538]],
            "lighting": "균일 조명 (실내 형광등)",
            "angle_deg": 4,
            "background_desc": "어두운 단색 책상",
            "notes": "마트 영수증 정면 촬영",
        },
        {
            "id": "easy_03",
            "condition": "단순 배경 + 정면",
            "difficulty": "easy",
            "doc_type": "medical_form",
            "doc_size": (410, 510),
            "bg": "dark_matte",
            "corners": [[185, 62], [605, 52], [615, 540], [195, 548]],
            "lighting": "균일 조명",
            "angle_deg": 6,
            "background_desc": "단순 회색 매트",
            "notes": "의료 처방전 미세 기울기",
        },
        {
            "id": "easy_04",
            "condition": "단순 배경 + 정면",
            "difficulty": "easy",
            "doc_type": "business_card",
            "doc_size": (460, 300),
            "bg": "dark_wood",
            "corners": [[170, 145], [630, 152], [625, 452], [166, 445]],
            "lighting": "균일 조명",
            "angle_deg": 3,
            "background_desc": "어두운 원목 테이블",
            "notes": "가로형 명함 정면",
        },
        {
            "id": "easy_05",
            "condition": "단순 배경 + 정면",
            "difficulty": "easy",
            "doc_type": "invoice",
            "doc_size": (400, 520),
            "bg": "dark_matte",
            "corners": [[212, 58], [595, 68], [588, 542], [202, 534]],
            "lighting": "균일 조명",
            "angle_deg": 7,
            "background_desc": "어두운 무광 패드",
            "notes": "상업 송장 정면",
        },
        # 2. 그림자 있음 (5장, hard)
        {
            "id": "shadow_01",
            "condition": "그림자 있음",
            "difficulty": "hard",
            "doc_type": "a4_report",
            "doc_size": (400, 520),
            "bg": "dark_matte",
            "corners": [[195, 60], [605, 66], [608, 544], [192, 538]],
            "lighting": "부드러운 대각선 그림자 (35% 감광)",
            "angle_deg": 5,
            "background_desc": "어두운 매트",
            "effect": "soft_shadow",
            "notes": "연한 그림자 — 외곽선 유지 가능",
        },
        {
            "id": "shadow_02",
            "condition": "그림자 있음",
            "difficulty": "hard",
            "doc_type": "receipt",
            "doc_size": (330, 520),
            "bg": "dark_wood",
            "corners": [[235, 58], [562, 62], [568, 540], [230, 536]],
            "lighting": "중간 그림자 + 약한 저조도",
            "angle_deg": 8,
            "background_desc": "어두운 책상",
            "effect": "medium_shadow",
            "notes": "우하단 모서리에 드리워진 그림자",
        },
        {
            "id": "shadow_03",
            "condition": "그림자 있음",
            "difficulty": "hard",
            "doc_type": "invoice",
            "doc_size": (400, 520),
            "bg": "dark_matte",
            "corners": [[195, 56], [600, 62], [604, 542], [190, 538]],
            "lighting": "강한 경계 그림자 (75% 감광, 날카로운 음영 경계)",
            "angle_deg": 6,
            "background_desc": "어두운 매트",
            "effect": "hard_edge_shadow",
            "notes": "날카로운 그림자 경계가 가짜 엣지를 형성",
        },
        {
            "id": "shadow_04",
            "condition": "그림자 있음",
            "difficulty": "hard",
            "doc_type": "medical_form",
            "doc_size": (400, 520),
            "bg": "dark_matte",
            "corners": [[200, 60], [598, 66], [602, 540], [196, 535]],
            "lighting": "심각한 저조도 + 82% 암부 그림자",
            "angle_deg": 9,
            "background_desc": "어두운 매트",
            "effect": "extreme_dark_shadow",
            "notes": "어두운 영역 밝기 압축으로 엣지 기울기 급감",
        },
        {
            "id": "shadow_05",
            "condition": "그림자 있음",
            "difficulty": "hard",
            "doc_type": "a4_report",
            "doc_size": (400, 520),
            "bg": "beige_low_contrast",
            "corners": [[195, 62], [602, 65], [600, 538], [192, 535]],
            "lighting": "강한 그림자 + 우상단 스포트라이트 빛 반사(Glare)",
            "angle_deg": 10,
            "background_desc": "밝은 책상 + 조명 반사",
            "effect": "shadow_plus_glare",
            "notes": "빛 반사로 모서리 소실 + 그림자 엣지 간섭",
        },
        # 3. 45도 이상 기울어짐 (5장, hard)
        {
            "id": "tilt_01",
            "condition": "45도 이상 기울어짐",
            "difficulty": "hard",
            "doc_type": "a4_report",
            "doc_size": (380, 480),
            "bg": "dark_matte",
            "corners": [[395, 58], [662, 265], [412, 542], [142, 332]],
            "lighting": "균일 조명",
            "angle_deg": 46,
            "background_desc": "어두운 매트",
            "effect": "tilt_clean",
            "notes": "46도 마름모 회전 — 단순 sum/diff 정렬 실패 테스트 케이스",
        },
        {
            "id": "tilt_02",
            "condition": "45도 이상 기울어짐",
            "difficulty": "hard",
            "doc_type": "invoice",
            "doc_size": (380, 500),
            "bg": "dark_wood",
            "corners": [[260, 88], [555, 118], [675, 522], [125, 488]],
            "lighting": "균일 조명",
            "angle_deg": 50,
            "background_desc": "어두운 원목",
            "effect": "tilt_clean",
            "notes": "50도 원근 사다리꼴 왜곡",
        },
        {
            "id": "tilt_03",
            "condition": "45도 이상 기울어짐",
            "difficulty": "hard",
            "doc_type": "receipt",
            "doc_size": (320, 500),
            "bg": "dark_matte",
            "corners": [[285, 105], [520, 128], [685, 515], [115, 492]],
            "lighting": "원거리 엣지 미세 블러 + 55도 기울기",
            "angle_deg": 55,
            "background_desc": "어두운 매트",
            "effect": "tilt_far_blur",
            "notes": "원거리 초점 흐림으로 상단 엣지 약화",
        },
        {
            "id": "tilt_04",
            "condition": "45도 이상 기울어짐",
            "difficulty": "hard",
            "doc_type": "medical_form",
            "doc_size": (380, 500),
            "bg": "wood_clutter",
            "corners": [[310, 138], [520, 152], [728, 525], [88, 505]],
            "lighting": "60도 급경사 + 원거리 음영",
            "angle_deg": 60,
            "background_desc": "원목 + 측면 물체",
            "effect": "tilt_extreme_dim",
            "notes": "60도 극심한 원근 축소 + 상단부 저대비",
        },
        {
            "id": "tilt_05",
            "condition": "45도 이상 기울어짐",
            "difficulty": "hard",
            "doc_type": "a4_report",
            "doc_size": (380, 500),
            "bg": "dark_wood",
            "corners": [[318, 162], [512, 178], [742, 518], [72, 495]],
            "lighting": "65도 극심한 기울기 + 종이 말림(곡선 왜곡)",
            "angle_deg": 65,
            "background_desc": "어두운 책상",
            "effect": "tilt_curled_edge",
            "notes": "상단 가장자리 말림으로 4꼭짓점 다각형 근사 붕괴",
        },
        # 4. 복잡한 배경 (5장, hard)
        {
            "id": "complex_01",
            "condition": "복잡한 배경",
            "difficulty": "hard",
            "doc_type": "a4_report",
            "doc_size": (390, 500),
            "bg": "wood_clutter",
            "corners": [[205, 62], [595, 68], [592, 535], [200, 530]],
            "lighting": "균일 조명",
            "angle_deg": 5,
            "background_desc": "나뭇결 책상 + 양옆 소품",
            "effect": "none",
            "notes": "배경 잡동사니 존재하지만 문서 대비는 충분",
        },
        {
            "id": "complex_02",
            "condition": "복잡한 배경",
            "difficulty": "hard",
            "doc_type": "invoice",
            "doc_size": (390, 500),
            "bg": "grid_plaid",
            "corners": [[202, 66], [592, 72], [588, 532], [198, 526]],
            "lighting": "균일 조명",
            "angle_deg": 6,
            "background_desc": "강한 격자무늬 매트 + 볼펜",
            "effect": "grid_interference",
            "notes": "격자 패턴이 미세 엣지 노이즈를 대량 유발",
        },
        {
            "id": "complex_03",
            "condition": "복잡한 배경",
            "difficulty": "hard",
            "doc_type": "receipt",
            "doc_size": (340, 500),
            "bg": "beige_low_contrast",
            "corners": [[228, 62], [568, 66], [570, 536], [225, 532]],
            "lighting": "밝은 베이지 배경 (종이와 유사 명도)",
            "angle_deg": 4,
            "background_desc": "흰 종이와 명도 차이가 작은 베이지 책상",
            "effect": "low_contrast_paper",
            "notes": "배경-문서 휘도차 극소화로 외곽 Canny 미검출",
        },
        {
            "id": "complex_04",
            "condition": "복잡한 배경",
            "difficulty": "hard",
            "doc_type": "medical_form",
            "doc_size": (390, 500),
            "bg": "cluttered_desk",
            "corners": [[206, 65], [592, 70], [588, 532], [202, 528]],
            "lighting": "구겨진 종이 주름 + 복잡한 책상 패턴",
            "angle_deg": 8,
            "background_desc": "복잡한 도형/선분 난무 배경 + 종이 구겨짐",
            "effect": "crumpled_and_clutter",
            "notes": "내부 구김 주름 + 배경 교차선으로 윤곽선 분절",
        },
        {
            "id": "complex_05",
            "condition": "복잡한 배경",
            "difficulty": "hard",
            "doc_type": "a4_report",
            "doc_size": (390, 500),
            "bg": "cluttered_desk",
            "corners": [[205, 64], [594, 68], [590, 534], [200, 530]],
            "lighting": "우하단 손가락/물체 부분 가림 + 복잡한 배경",
            "angle_deg": 5,
            "background_desc": "복잡한 배경 + 모서리 부분 가림(Occlusion)",
            "effect": "corner_occlusion",
            "notes": "모서리 가림으로 사각형이 6각형 L자 윤곽으로 변형",
        },
    ]

    metadata_records: List[Dict[str, Any]] = []

    for spec in specs:
        paper_col = (224, 228, 232) if spec.get("effect") == "low_contrast_paper" else (244, 245, 247)
        doc_w, doc_h = spec["doc_size"]
        doc_tex = _create_document_texture(spec["doc_type"], width=doc_w, height=doc_h, paper_color=paper_col)
        bg = _make_background(spec["bg"], rng=rng)
        corners = np.asarray(spec["corners"], dtype=np.float32)

        scene = _composite_document(bg, doc_tex, corners)
        effect = spec.get("effect", "none")

        if effect == "soft_shadow":
            scene = _apply_diagonal_shadow(scene, strength=0.32, blur_ksize=65, slope=0.55, offset=150)
        elif effect == "medium_shadow":
            scene = _apply_diagonal_shadow(scene, strength=0.48, blur_ksize=41, slope=0.72, offset=110)
            scene = np.clip(scene.astype(np.float32) * 0.88, 0, 255).astype(np.uint8)
        elif effect == "hard_edge_shadow":
            scene = _apply_diagonal_shadow(scene, strength=0.78, blur_ksize=3, slope=0.82, offset=45)
        elif effect == "extreme_dark_shadow":
            scene = np.clip(scene.astype(np.float32) * 0.34, 0, 255).astype(np.uint8)
            scene = _apply_diagonal_shadow(scene, strength=0.72, blur_ksize=7, slope=-0.35, offset=360)
        elif effect == "shadow_plus_glare":
            scene = _apply_diagonal_shadow(scene, strength=0.68, blur_ksize=5, slope=0.75, offset=120)
            scene = _apply_spotlight_glare(scene, center=(585, 88), radius=105.0, intensity=130.0)
        elif effect == "tilt_far_blur":
            top_blur = cv2.GaussianBlur(scene, (11, 11), 0)
            h = scene.shape[0]
            alpha = np.clip((260.0 - np.arange(h, dtype=np.float32)) / 180.0, 0.0, 0.85)[:, None, None]
            scene = np.clip(scene * (1.0 - alpha) + top_blur * alpha, 0, 255).astype(np.uint8)
        elif effect == "tilt_extreme_dim":
            scene = _apply_diagonal_shadow(scene, strength=0.62, blur_ksize=9, slope=-0.2, offset=240)
            scene = cv2.GaussianBlur(scene, (5, 5), 0)
        elif effect == "tilt_curled_edge":
            # Cover top border with dark curved wave so outer edge is curved/broken
            pts_wave = np.array(
                [[310, 155], [375, 195], [445, 198], [520, 172], [520, 130], [310, 130]],
                dtype=np.int32,
            )
            cv2.fillPoly(scene, [pts_wave], (42, 54, 72))
            scene = _apply_diagonal_shadow(scene, strength=0.55, blur_ksize=11, slope=0.35, offset=190)
        elif effect == "grid_interference":
            # Subtle noise + slightly softer border contrast on busy plaid
            noise = rng.normal(0, 14.0, size=scene.shape)
            scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif effect == "low_contrast_paper":
            scene = cv2.GaussianBlur(scene, (3, 3), 0)
        elif effect == "crumpled_and_clutter":
            scene = _apply_crease_folds(scene, corners)
            # Draw overlapping dark cable across right edge
            cv2.line(scene, (520, 220), (760, 310), (35, 35, 40), 16)
            cv2.line(scene, (120, 390), (295, 430), (40, 45, 55), 14)
        elif effect == "corner_occlusion":
            # Occlude bottom-right corner with a realistic skin-tone thumb/finger + pen
            cv2.ellipse(scene, (575, 510), (75, 42), -25, 0, 360, (128, 154, 196), -1)
            cv2.ellipse(scene, (575, 510), (75, 42), -25, 0, 360, (95, 118, 155), 2)
            cv2.line(scene, (130, 110), (265, 175), (45, 45, 55), 18)

        # Mild realistic camera sensor noise
        sensor_noise = rng.normal(0, 2.2, size=scene.shape)
        scene = np.clip(scene.astype(np.float32) + sensor_noise, 0, 255).astype(np.uint8)

        filename = f"{spec['id']}.jpg"
        save_image(img_dir / filename, scene)

        metadata_records.append(
            {
                "id": spec["id"],
                "filename": filename,
                "relative_path": f"test_images/{filename}",
                "condition": spec["condition"],
                "difficulty": spec["difficulty"],
                "document_type": spec["doc_type"],
                "lighting": spec["lighting"],
                "angle_deg": spec["angle_deg"],
                "background": spec["background_desc"],
                "notes": spec["notes"],
                "gt_corners": spec["corners"],
            }
        )

    payload = {
        "dataset_name": "codyssey-a3-01-benchmark-20",
        "image_size": [800, 600],
        "total_images": len(metadata_records),
        "images": metadata_records,
    }
    save_json(root / "metadata.json", payload)
    return payload


if __name__ == "__main__":
    configure_utf8_console()
    result = generate_dataset()
    print(f"Generated {result['total_images']} benchmark images in data/test_images/ and data/metadata.json")
