"""Generate high-resolution Matplotlib diagrams for architecture, math, EDA, and benchmarks.

All generated PNGs are saved under `diagrams/` and embedded via repository-relative paths
(`diagrams/*.png`) to guarantee reliable rendering on GitHub and local Markdown viewers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PRESETS, ScanConfig
from src.geometry import order_points, order_points_naive_sum_diff
from src.io_utils import configure_utf8_console, load_image
from src.pipeline import DocumentScanner

DIAGRAMS_DIR = PROJECT_ROOT / "diagrams"


def _setup_korean_font() -> None:
    """Configure a clean Korean-capable font on Windows/macOS/Linux without glyph warnings."""
    plt.rcParams["axes.unicode_minus"] = False
    for candidate in ("Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"):
        try:
            plt.rcParams["font.family"] = candidate
            break
        except Exception:
            continue


def generate_01_architecture_pipeline() -> Path:
    """Diagram 1: SRP Module Architecture & 5-Stage Document Scanner Pipeline."""
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(
        7.5,
        8.05,
        "OpenCV & NumPy 지능형 문서 스캐너 — SRP 모듈 아키텍처 및 5단계 처리 파이프라인",
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        7.5,
        7.62,
        "Zero Deep Learning · Zero OCR · Zero Scipy/Shapely | 순수 규칙 기반 파이프라인 & 실시간 트랙바 / 웹캠 지원",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#475569",
    )

    stages = [
        ("Stage 0\n입력 & 안전 I/O", "src/io_utils.py\n• 한글 경로 np.fromfile\n• cv2.imdecode 로드\n• ScanConfig 검증", "#DBEAFE", "#2563EB"),
        ("Stage 1\n전처리 & 엣지", "src/preprocessor.py\n• Grayscale 변환\n• Gaussian Blur (k×k)\n• Canny Edge (low/high)", "#E0E7FF", "#4F46E5"),
        ("Stage 2\n윤곽선 & 사각형", "src/contour_detector.py\n• cv2.findContours\n• 면적 비율 필터링\n• approxPolyDP (4꼭짓점)", "#FEF3C7", "#D97706"),
        ("Stage 3\n수동 정렬 & 워핑", "src/geometry.py\n• 무게중심 편각 정렬\n• [TL, TR, BR, BL] 보장\n• warpPerspective 변환", "#DCFCE7", "#16A34A"),
        ("Stage 4\n스캔 후처리", "src/postprocessor.py\n• Adaptive Threshold\n• 조명/그림자 보정\n• 한글 안전 imencode", "#FCE7F3", "#DB2777"),
    ]

    box_w = 2.48
    box_h = 2.35
    y_top = 4.65
    xs = [0.45, 3.35, 6.25, 9.15, 12.05]

    for idx, (title, body, fill, edge) in enumerate(stages):
        x = xs[idx]
        rect = patches.FancyBboxPatch(
            (x, y_top),
            box_w,
            box_h,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor=fill,
            edgecolor=edge,
            linewidth=2.2,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y_top + box_h - 0.42, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#0F172A")
        ax.plot([x + 0.15, x + box_w - 0.15], [y_top + box_h - 0.85, y_top + box_h - 0.85], color=edge, linewidth=1.2, alpha=0.6)
        ax.text(x + 0.16, y_top + 0.72, body, ha="left", va="center", fontsize=9.0, color="#1E293B", linespacing=1.45)

        if idx < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.06, y_top + box_h / 2),
                xytext=(x + box_w + 0.06, y_top + box_h / 2),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", color="#334155", lw=2.2),
            )

    # Bottom orchestration & interface layer
    bottom_boxes = [
        (0.45, 1.35, 4.35, 2.15, "오케스트레이터 & CLI 진입점", "main.py / src/cli.py / src/pipeline.py\n• 단일 이미지 즉시 스캔 (python main.py img.jpg)\n• 6패널 단계별 몽타주 자동 생성 (헤드리스/Colab 대응)\n• 실패 시 크래시 없이 failure_reason 기록", "#F1F5F9", "#475569"),
        (5.25, 1.35, 4.45, 2.15, "실시간 GUI 트랙바 & 웹캠 모드", "src/gui.py\n• 5단계 실시간 OpenCV 윈도우 시각화\n• 트랙바: Blur Kernel / Canny Low / High / Morph\n• 웹캠 모드: 실시간 윤곽선 + 's' 저장 / 'q' 종료", "#ECFEFF", "#0891B2"),
        (10.15, 1.35, 4.38, 2.15, "EDA & 20장 정량 평가 / 스윕", "src/eda.py / src/evaluator.py\n• 20장 픽셀 밝기 히스토그램 & Sobel 기울기 분석\n• 4조건×5장 IoU/꼭짓점 오차 정량 평가\n• 5개 프리셋 스윕으로 파라미터 딜레마 증명", "#FEF2F2", "#DC2626"),
    ]

    for x, y, w, h, title, body, fill, edge in bottom_boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.16",
            facecolor=fill,
            edgecolor=edge,
            linewidth=2.0,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", fontsize=10.8, fontweight="bold", color="#0F172A")
        ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.62, y + h - 0.62], color=edge, linewidth=1.1, alpha=0.55)
        ax.text(x + 0.18, y + 0.72, body, ha="left", va="center", fontsize=8.8, color="#1E293B", linespacing=1.45)

    # Vertical connectors linking pipeline to bottom subsystems
    for bx in (2.6, 7.5, 12.3):
        ax.annotate(
            "",
            xy=(bx, 4.55),
            xytext=(bx, 3.60),
            arrowprops=dict(arrowstyle="<->,head_width=0.28,head_length=0.45", color="#64748B", lw=1.7, ls="--"),
        )

    ax.text(
        7.5,
        0.55,
        "데이터 흐름: BGR (H, W, 3) → Gray (H, W) → Edge {0, 255} → Quad (4, 2) → Homography (3, 3) → Binary Scan {0, 255}",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#E2E8F0", edgecolor="#94A3B8"),
    )

    out = DIAGRAMS_DIR / "01_architecture_pipeline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_02_geometry_perspective_math() -> Path:
    """Diagram 2: Manual 4-Point Ordering (45-deg Diamond Tie-Break) & Homography Warp."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "수동 4점 좌표 정렬(무게중심 편각 방식)의 동률 방어 원리 및 투시 변환(Homography)",
        fontsize=14,
        fontweight="bold",
        color="#0F172A",
        y=1.01,
    )

    diamond = np.array([[100, 10], [190, 100], [100, 190], [10, 100]], dtype=np.float32)
    naive = order_points_naive_sum_diff(diamond)
    ours = order_points(diamond)

    # Panel 1: Naive sum/diff failure
    ax = axes[0]
    ax.set_title("① 기존 단순 sum/diff 방식 (45° 동률 붕괴!)", fontsize=11, fontweight="bold", color="#DC2626", pad=10)
    poly_true = np.vstack([diamond, diamond[0]])
    ax.plot(poly_true[:, 0], poly_true[:, 1], "--", color="#94A3B8", lw=1.5, label="실제 문서 외곽")
    poly_naive = np.vstack([naive, naive[0]])
    ax.fill(poly_naive[:, 0], poly_naive[:, 1], color="#FCA5A5", alpha=0.45)
    ax.plot(poly_naive[:, 0], poly_naive[:, 1], "-o", color="#DC2626", lw=2.5, ms=8, label="선택된 4점 경로")
    for idx, (px, py) in enumerate(naive):
        ax.text(px + 6, py - 8 - idx * 8, f"P{idx}({int(px)},{int(py)})", fontsize=8.5, color="#991B1B", fontweight="bold")
    ax.text(
        100,
        215,
        "x+y 동률(110=110)로 꼭짓점 중복!\n고유 꼭짓점 2~3개만 선택 → 워핑 붕괴",
        ha="center",
        fontsize=9,
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#F87171"),
    )
    ax.set_xlim(-15, 225)
    ax.set_ylim(235, -15)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: Our Polar Angle Ordering
    ax = axes[1]
    ax.set_title("② 본 프로젝트 무게중심 편각 정렬 (100% 방어)", fontsize=11, fontweight="bold", color="#16A34A", pad=10)
    poly_ours = np.vstack([ours, ours[0]])
    ax.fill(poly_ours[:, 0], poly_ours[:, 1], color="#BBF7D0", alpha=0.5)
    ax.plot(poly_ours[:, 0], poly_ours[:, 1], "-o", color="#15803D", lw=2.5, ms=8, label="시계방향 [TL→TR→BR→BL]")
    cx, cy = float(np.mean(ours[:, 0])), float(np.mean(ours[:, 1]))
    ax.scatter([cx], [cy], color="#2563EB", s=70, zorder=5, label="무게중심 (cx, cy)")
    labels = ["0: TL", "1: TR", "2: BR", "3: BL"]
    offsets = [(10, -4), (8, -6), (12, 4), (-52, 14)]
    for i, ((px, py), lbl, (dx, dy)) in enumerate(zip(ours, labels, offsets)):
        ax.annotate("", xy=(px, py), xytext=(cx, cy), arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.3, alpha=0.7))
        ax.text(px + dx, py + dy, lbl, fontsize=9, fontweight="bold", color="#14532D")
    ax.text(
        100,
        222,
        "atan2(y-cy, x-cx) 시계방향 정렬 +\n사전순(x+y, y, x) TL 고정 → 중복 0건",
        ha="center",
        fontsize=9,
        color="#14532D",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0FDF4", edgecolor="#4ADE80"),
    )
    ax.set_xlim(-15, 225)
    ax.set_ylim(246, -22)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 3: Perspective Transformation (Homography)
    ax = axes[2]
    ax.set_title("③ 3×3 투시 변환 행렬 (getPerspectiveTransform)", fontsize=11, fontweight="bold", color="#1D4ED8", pad=10)
    src_q = np.array([[35, 45], [105, 35], [120, 145], [15, 130]], dtype=np.float32)
    dst_q = np.array([[155, 35], [225, 35], [225, 145], [155, 145]], dtype=np.float32)

    ax.fill(src_q[:, 0], src_q[:, 1], color="#DBEAFE", edgecolor="#2563EB", lw=2)
    ax.fill(dst_q[:, 0], dst_q[:, 1], color="#FEF3C7", edgecolor="#D97706", lw=2)
    for sp, dp, lbl in zip(src_q, dst_q, ["TL", "TR", "BR", "BL"]):
        ax.annotate("", xy=(dp[0] - 2, dp[1]), xytext=(sp[0] + 2, sp[1]), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.1, ls=":"))
        ax.text(sp[0] - 12, sp[1] - 6, lbl, fontsize=8, color="#1E40AF", fontweight="bold")
        ax.text(dp[0] + 3, dp[1] - 6, f"{lbl}'", fontsize=8, color="#92400E", fontweight="bold")

    ax.text(68, 92, "기울어진\n원본 문서", ha="center", va="center", fontsize=9, fontweight="bold", color="#1E3A8A")
    ax.text(190, 92, "정면 스캔\n직사각형\n(W × H)", ha="center", va="center", fontsize=9, fontweight="bold", color="#92400E")
    ax.text(
        120,
        205,
        "W = max(||TR-TL||, ||BR-BL||)\nH = max(||TR-BR||, ||TL-BL||)\n[x', y', w']^T = M_(3×3) · [x, y, 1]^T",
        ha="center",
        fontsize=8.8,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#EFF6FF", edgecolor="#60A5FA"),
    )
    ax.set_xlim(-5, 250)
    ax.set_ylim(235, -10)
    ax.grid(True, alpha=0.25)

    out = DIAGRAMS_DIR / "02_geometry_perspective_math.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_03_eda_histograms() -> Path:
    """Diagram 3: Real 20-image Flow Photo Brightness Histogram Comparison & Dark-Image Analysis."""
    eda = json.loads((PROJECT_ROOT / "outputs" / "eda_summary.json").read_text(encoding="utf-8"))
    bins = np.arange(256)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.6), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "Google Flow 실사 20장 이미지 탐색적 분석(EDA) — 밝기 히스토그램 및 조건별 대비/기울기 특성",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
    )

    # Left: Normalized histograms per condition
    ax = axes[0]
    palette = {
        "단순 배경 + 정면": ("#16A34A", 2.4, "-"),
        "그림자 있음": ("#DC2626", 2.2, "-"),
        "45도 이상 기울어짐": ("#2563EB", 2.0, "--"),
        "복잡한 배경": ("#D97706", 2.0, "-."),
    }
    ax.axvspan(0, 60, color="#FEE2E2", alpha=0.45, label="암부 압축 구간 (0~60: 기울기 저하)")
    max_y = 0.0
    for cond, (color, lw, ls) in palette.items():
        hist = np.array(eda["condition_summary"][cond]["mean_histogram"])
        smooth = np.convolve(hist, np.ones(5) / 5.0, mode="same")
        max_y = max(max_y, float(smooth.max()))
        ax.plot(bins, smooth, color=color, linewidth=lw, linestyle=ls, label=f"{cond} (n=5)")

    top_y = max(0.030, max_y * 1.28)
    ax.annotate(
        "쉬운 케이스: 어두운 배경과 밝은 종이(210~240)의\n뚜렷한 양봉(Bimodal) 분리 → 엣지 선명",
        xy=(225, max_y * 0.85),
        xytext=(92, top_y * 0.80),
        fontsize=8.5,
        fontweight="bold",
        color="#14532D",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor="#4ADE80"),
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.6),
    )
    ax.annotate(
        "그림자 케이스: 암부(0~55)로 픽셀 쏠림\n→ 밝기차(ΔI) 축소로 Canny 외곽 단절",
        xy=(24, max_y * 0.48),
        xytext=(58, top_y * 0.56),
        fontsize=8.4,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#F87171"),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.6),
    )
    ax.set_title("① 조건별 평균 픽셀 밝기 분포 (256-Bin Normalized Histogram)", fontsize=11, fontweight="bold")
    ax.set_xlabel("픽셀 밝기 값 (0 = 완전 검정, 255 = 완전 흰색)", fontsize=10)
    ax.set_ylabel("정규화 픽셀 비율", fontsize=10)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, top_y)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.3, framealpha=0.92)

    # Right: Key statistical metrics per condition
    ax2 = axes[1]
    conds = ["단순 배경 + 정면", "그림자 있음", "45도 이상 기울어짐", "복잡한 배경"]
    short_labels = ["단순 배경\n+ 정면", "그림자\n있음", "45° 이상\n기울어짐", "복잡한\n배경"]
    stds = [eda["condition_summary"][c]["std_brightness"] for c in conds]
    grads = [eda["condition_summary"][c]["mean_gradient"] for c in conds]
    dyn = [eda["condition_summary"][c]["dynamic_range"] for c in conds]

    x = np.arange(len(conds))
    w = 0.26
    b1 = ax2.bar(x - w, stds, width=w, color="#2563EB", label="밝기 표준편차 (Std)")
    b2 = ax2.bar(x, grads, width=w, color="#D97706", label="평균 Sobel 기울기 (|∇I|)")
    b3 = ax2.bar(x + w, [d / 2.5 for d in dyn], width=w, color="#10B981", label="동적 범위 (p95-p05) ÷ 2.5")

    for bars in (b1, b2, b3):
        for bar in bars:
            val = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}", ha="center", va="bottom", fontsize=8.2)

    ax2.set_title("② 조건별 대비(Std)·기울기(|∇I|)·동적 범위 비교", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=9.5)
    ax2.set_ylabel("수치 (픽셀 강도 / 기울기 단위)", fontsize=10)
    ax2.set_ylim(0, 108)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.8)

    out = DIAGRAMS_DIR / "03_eda_brightness_histograms.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_04_benchmark_parameter_dilemma() -> Path:
    """Diagram 4: 20-Image x 5-Preset Heatmap & Condition Success Rate (Parameter Dilemma)."""
    sweep = json.loads((PROJECT_ROOT / "outputs" / "sweep_summary.json").read_text(encoding="utf-8"))

    preset_names = [
        "default",
        "strong_denoise",
        "morph_bridge",
        "clahe_shadow_morph",
        "heavy_blur_morph",
    ]
    preset_labels = [
        "default\n(k5, 50/150)",
        "strong_denoise\n(k9, 80/200+M3)",
        "morph_bridge\n(k5, 40/120+M5)",
        "clahe_shadow\n(k7, 70/180+M9)",
        "heavy_blur\n(k11, 20/60+M9)",
    ]
    image_ids = list(sweep["per_image_matrix"].keys())

    matrix = np.zeros((len(image_ids), len(preset_names)), dtype=np.int32)
    for r, img_id in enumerate(image_ids):
        for c, pname in enumerate(preset_names):
            matrix[r, c] = 1 if sweep["per_image_matrix"][img_id][pname] else 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6), dpi=180, gridspec_kw={"width_ratios": [1.22, 1.0]})
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "Google Flow 실사 20장 파라미터 딜레마(Parameter Dilemma) 실증 — 5개 프리셋 성공/실패 매트릭스",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
    )

    # Left: Heatmap
    ax = axes[0]
    cmap = matplotlib.colors.ListedColormap(["#FEE2E2", "#BBF7D0"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for r in range(len(image_ids)):
        for c in range(len(preset_names)):
            val = matrix[r, c]
            txt = "PASS" if val == 1 else "FAIL"
            col = "#14532D" if val == 1 else "#991B1B"
            ax.text(c, r, txt, ha="center", va="center", fontsize=8.2, fontweight="bold", color=col)

    # Highlight conflict rows
    for special_id, note_color in [("shadow_02", "#2563EB"), ("tilt_03", "#7C3AED"), ("complex_01", "#D97706")]:
        r_idx = image_ids.index(special_id)
        rect = patches.Rectangle((-0.48, r_idx - 0.48), len(preset_names) - 0.04, 0.96, fill=False, edgecolor=note_color, lw=2.3)
        ax.add_patch(rect)

    for boundary in (4.5, 9.5, 14.5):
        ax.axhline(boundary, color="#334155", linewidth=1.8)

    ax.set_xticks(np.arange(len(preset_names)))
    ax.set_xticklabels(preset_labels, fontsize=8.4)
    ax.set_yticks(np.arange(len(image_ids)))
    ax.set_yticklabels(image_ids, fontsize=8.8)
    ax.set_title("① 이미지별 × 프리셋별 성공 매트릭스 (강조 테두리 = 상충 사례)", fontsize=11, fontweight="bold", pad=10)

    # Right: Condition-wise grouped bar chart
    ax2 = axes[1]
    conds = ["단순 배경 + 정면", "그림자 있음", "45도 이상 기울어짐", "복잡한 배경"]
    short_conds = ["단순 배경\n+ 정면", "그림자\n있음", "45° 이상\n기울어짐", "복잡한\n배경", "전체 합계\n(20장)"]

    def _rates(preset_key: str) -> list[float]:
        by_c = {row["condition"]: row["success_rate"] for row in sweep["presets"][preset_key]["by_condition"]}
        return [by_c[c] for c in conds] + [sweep["presets"][preset_key]["overall"]["success_rate"]]

    oracle_rates = []
    for c_idx, cond in enumerate(conds):
        ids_in_cond = image_ids[c_idx * 5 : (c_idx + 1) * 5]
        ok = sum(1 for i in ids_in_cond if any(sweep["per_image_matrix"][i].values()))
        oracle_rates.append(ok / 5.0 * 100.0)
    oracle_rates.append(sweep["oracle_union_success"] / 20.0 * 100.0)

    def_pct = sweep["presets"]["default"]["overall"]["success_rate"]
    mb_pct = sweep["presets"]["morph_bridge"]["overall"]["success_rate"]
    cs_pct = sweep["presets"]["clahe_shadow_morph"]["overall"]["success_rate"]
    ora_pct = sweep["oracle_union_success"] / 20.0 * 100.0

    x = np.arange(len(short_conds))
    w = 0.2
    series = [
        (f"default ({def_pct:.0f}%)", _rates("default"), "#64748B"),
        (f"morph_bridge ({mb_pct:.0f}%)", _rates("morph_bridge"), "#10B981"),
        (f"clahe_shadow ({cs_pct:.0f}%)", _rates("clahe_shadow_morph"), "#2563EB"),
        (f"Oracle 합집합 ({ora_pct:.0f}%)", oracle_rates, "#F59E0B"),
    ]
    for i, (lbl, vals, col) in enumerate(series):
        bars = ax2.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=col, label=lbl)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=6.7)

    ax2.set_title(
        f"② 조건별 성공률 비교 — 단일 한계({sweep['best_single_preset_success']}/20) vs Oracle({sweep['oracle_union_success']}/20)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_conds, fontsize=9.2)
    ax2.set_ylabel("검출 성공률 (%)", fontsize=10)
    ax2.set_ylim(0, 118)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.5)

    hard_fail = 20 - sweep["oracle_union_success"]
    ax2.text(
        2.0,
        -22,
        "핵심 결론: clahe_shadow(shadow_02 구조) ↔ heavy_blur(tilt_03 구조) ↔ morph_bridge(complex_01 구조)\n"
        f"서로 다른 실사 사진이 정반대 파라미터를 요구하며, {hard_fail}장({hard_fail*5}%)은 어떤 규칙 조합으로도 해결 불가!",
        ha="center",
        va="top",
        fontsize=8.8,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFBEB", edgecolor="#F59E0B"),
    )

    out = DIAGRAMS_DIR / "04_benchmark_parameter_dilemma.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_05_real_pipeline_examples() -> Path:
    """Diagram 5: Real visual side-by-side results on Google Flow photorealistic photos."""
    scanner = DocumentScanner()
    cases = [
        ("easy_01.jpg", PRESETS["default"], "① easy_01\n(단순+정면 / Default)"),
        ("shadow_02.jpg", PRESETS["clahe_shadow_morph"], "② shadow_02\n(손 그림자 / CLAHE 구조)"),
        ("tilt_03.jpg", PRESETS["heavy_blur_morph"], "③ tilt_03\n(55° 영수증 / 강블러 구조)"),
        ("complex_05.jpg", PRESETS["default"], "④ complex_05\n(손가락+펜 가림 실패)"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(15.0, 13.0), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "Google Flow 실사 20장 파이프라인 실제 실행 결과 (원본 → Canny 엣지 → 윤곽선 검출 → 스캔 결과)",
        fontsize=14,
        fontweight="bold",
        color="#0F172A",
    )

    col_titles = ["1. 입력 실사 원본", "2. Canny 엣지", "3. 윤곽선/꼭짓점 오버레이", "4. 최종 적응형 이진화 스캔"]

    for row_idx, (fname, cfg, row_label) in enumerate(cases):
        img = load_image(PROJECT_ROOT / "data" / "flow_document_images" / fname)
        res = scanner.scan_image(img, cfg)

        panels = [
            cv2.cvtColor(res.original, cv2.COLOR_BGR2RGB),
            res.edges,
            cv2.cvtColor(res.contour_overlay, cv2.COLOR_BGR2RGB),
            res.scanned,
        ]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10.5, fontweight="bold", pad=8)
            if panel is None:
                ax.set_facecolor("#FEF2F2")
                ax.text(
                    0.5,
                    0.5,
                    f"검출 실패\n({res.failure_reason})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#DC2626",
                    transform=ax.transAxes,
                )
            elif panel.ndim == 2:
                ax.imshow(panel, cmap="gray")
            else:
                ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=8.8, fontweight="bold", color="#1E293B", rotation=90, labelpad=8)

    out = DIAGRAMS_DIR / "05_real_pipeline_examples.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_06_contour_approx_guide() -> Path:
    """Diagram 6: Visual 4-step beginner guide to findContours, Area Filter, approxPolyDP, and Convexity."""
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 5.2), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "3단계 완전 시각화: 윤곽선 탐색(findContours) → 면적 필터링 → 다각형 근사(approxPolyDP) → 볼록성 검사",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
        y=1.02,
    )

    # Panel 1: findContours raw candidates
    ax = axes[0]
    ax.set_title("① findContours\n(모든 닫힌 도형 수집)", fontsize=10.5, fontweight="bold", color="#1E293B", pad=8)
    ax.set_facecolor("#0F172A")
    # Outer screen frame (99%)
    ax.add_patch(patches.Rectangle((3, 3), 94, 94, fill=False, edgecolor="#94A3B8", lw=2.0, ls="--"))
    ax.text(50, 92, "화면 전체 테두리 (99%)", ha="center", va="center", fontsize=8, color="#CBD5E1", fontweight="bold")
    # Document candidate (42%)
    doc_pts = np.array([[22, 18], [78, 22], [82, 78], [18, 74], [22, 18]])
    ax.plot(doc_pts[:, 0], doc_pts[:, 1], color="#38BDF8", lw=2.5)
    ax.text(50, 49, "문서 후보\n(면적 42%)", ha="center", va="center", fontsize=9, color="#38BDF8", fontweight="bold")
    # Small dust & text specks (<5%)
    for (sx, sy, sw, sh) in [(8, 10, 7, 5), (84, 82, 8, 6), (34, 60, 12, 4), (52, 34, 10, 4)]:
        ax.add_patch(patches.Rectangle((sx, sy), sw, sh, fill=False, edgecolor="#FBBF24", lw=1.6))
    ax.text(28, 10, "먼지/글자 조각 (0.5%)", ha="left", va="center", fontsize=7.8, color="#FDE68A", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 2: Area Filter (5% ~ 98%)
    ax = axes[1]
    ax.set_title("② 면적 필터링 (5% ~ 98%)\n(너무 작거나 큰 후보 제거)", fontsize=10.5, fontweight="bold", color="#D97706", pad=8)
    ax.set_facecolor("#0F172A")
    ax.add_patch(patches.Rectangle((3, 3), 94, 94, fill=False, edgecolor="#EF4444", lw=1.8, ls=":"))
    ax.text(50, 92, "× 99% > 98% (프레임 탈락)", ha="center", va="center", fontsize=8, color="#FCA5A5", fontweight="bold")
    ax.fill(doc_pts[:, 0], doc_pts[:, 1], color="#10B981", alpha=0.22)
    ax.plot(doc_pts[:, 0], doc_pts[:, 1], color="#34D399", lw=2.8)
    ax.text(50, 49, "○ 42% 통과!\n(5% ≤ 42% ≤ 98%)", ha="center", va="center", fontsize=9, color="#6EE7B7", fontweight="bold")
    for (sx, sy, sw, sh) in [(8, 10, 7, 5), (84, 82, 8, 6)]:
        ax.add_patch(patches.Rectangle((sx, sy), sw, sh, fill=False, edgecolor="#EF4444", lw=1.5))
        ax.text(sx + sw / 2, sy + sh / 2, "×", ha="center", va="center", fontsize=11, color="#EF4444", fontweight="bold")
    ax.text(50, 10, "× 0.5% < 5% (티끌 탈락)", ha="center", va="center", fontsize=8, color="#FCA5A5", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 3: approxPolyDP (120 wobbly points -> 4 clean corners)
    ax = axes[2]
    ax.set_title("③ approxPolyDP (다각형 근사)\n(울퉁불퉁 점 120개 → 꼭짓점 4개)", fontsize=10.5, fontweight="bold", color="#2563EB", pad=8)
    ax.set_facecolor("#F8FAFC")
    rng = np.random.default_rng(7)
    base_corners = np.array([[20, 22], [80, 18], [84, 80], [16, 76]], dtype=float)
    wobbly_x, wobbly_y = [], []
    for i in range(4):
        p1, p2 = base_corners[i], base_corners[(i + 1) % 4]
        t = np.linspace(0, 1, 24, endpoint=False)
        edge_x = p1[0] + (p2[0] - p1[0]) * t + rng.uniform(-2.6, 2.6, size=len(t))
        edge_y = p1[1] + (p2[1] - p1[1]) * t + rng.uniform(-2.6, 2.6, size=len(t))
        wobbly_x.extend(edge_x)
        wobbly_y.extend(edge_y)
    wobbly_x.append(wobbly_x[0])
    wobbly_y.append(wobbly_y[0])

    ax.plot(wobbly_x, wobbly_y, color="#94A3B8", lw=1.5, label="원본 윤곽(흔들린 점 96개)")
    ax.scatter(wobbly_x[::3], wobbly_y[::3], color="#EF4444", s=12, zorder=3)
    clean_poly = np.vstack([base_corners, base_corners[0]])
    ax.plot(clean_poly[:, 0], clean_poly[:, 1], color="#16A34A", lw=2.8, label="approxPolyDP 직선화 (4점)")
    ax.scatter(base_corners[:, 0], base_corners[:, 1], color="#15803D", s=65, zorder=5)
    for idx, (cx, cy) in enumerate(base_corners):
        dx = -10 if cx < 50 else 5
        dy = -7 if cy < 50 else 5
        ax.text(cx + dx, cy + dy, f"V{idx+1}", fontsize=8.5, fontweight="bold", color="#14532D")
    ax.text(
        50,
        50,
        "허용 오차(ε=2%) 안의\n잔떨림을 쫙 펴서\n꼭짓점 4개만 남김!",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#1E3A8A",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#60A5FA"),
    )
    ax.set_xlim(2, 98)
    ax.set_ylim(2, 98)
    ax.legend(loc="lower center", fontsize=7.4, framealpha=0.92)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 4: Convex (PASS) vs Concave (FAIL)
    ax = axes[3]
    ax.set_title("④ 볼록(Convex) 최종 검사\n(정상 종이 ○ vs 움푹 파임 ×)", fontsize=10.5, fontweight="bold", color="#16A34A", pad=8)
    ax.set_facecolor("#F8FAFC")

    # Left: Convex PASS
    good_q = np.array([[8, 30], [42, 24], [46, 76], [10, 72], [8, 30]])
    ax.fill(good_q[:, 0], good_q[:, 1], color="#BBF7D0", alpha=0.65)
    ax.plot(good_q[:, 0], good_q[:, 1], "-o", color="#15803D", lw=2.3, ms=5)
    ax.text(27, 86, "○ 볼록 4각형\n(합격! PASS)", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#14532D")
    ax.text(27, 13, "고무줄로 감싸면\n4변이 팽팽함", ha="center", va="center", fontsize=7.8, color="#166534")

    ax.axvline(51, color="#CBD5E1", ls="--", lw=1.4)

    # Right: Concave / Occluded FAIL
    bad_q = np.array([[57, 26], [92, 26], [92, 56], [76, 56], [76, 74], [57, 74], [57, 26]])
    ax.fill(bad_q[:, 0], bad_q[:, 1], color="#FECACA", alpha=0.65)
    ax.plot(bad_q[:, 0], bad_q[:, 1], "-o", color="#DC2626", lw=2.3, ms=5)
    ax.annotate(
        "손가락 가림\n(움푹 파임!)",
        xy=(77, 57),
        xytext=(78, 78),
        ha="center",
        fontsize=7.6,
        fontweight="bold",
        color="#991B1B",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
    )
    ax.text(74, 13, "× 오목/6각형\n(탈락! FAIL)", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#991B1B")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])

    out = DIAGRAMS_DIR / "06_contour_approx_convex_guide.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_07_stage4_perspective_guide() -> Path:
    """Diagram 7: Visual 3-step beginner guide to Stage 4 (Edge length -> 3x3 Homography grid -> Real warp)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.4), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "4단계 완전 시각화: 변 길이 측정 → 3×3 투시 변환 행렬(격자 다림질) → 실제 문서 정면 복원",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
        y=1.02,
    )

    # Panel 1: Measuring 4 sides & choosing max W, max H
    ax = axes[0]
    ax.set_title("① 네 변 길이 재기 (피타고라스)\n(더 긴 변을 최종 가로·세로로 선택)", fontsize=10.5, fontweight="bold", color="#1E293B", pad=8)
    ax.set_facecolor("#F8FAFC")
    trap = np.array([[28, 78], [72, 74], [88, 20], [12, 24]], dtype=float)
    poly = np.vstack([trap, trap[0]])
    ax.fill(poly[:, 0], poly[:, 1], color="#DBEAFE", alpha=0.65)
    ax.plot(poly[:, 0], poly[:, 1], "-o", color="#2563EB", lw=2.5, ms=7)
    for (px, py), lbl, (dx, dy) in zip(trap, ["TL", "TR", "BR", "BL"], [(-11, 4), (4, 4), (3, -6), (-11, -6)]):
        ax.text(px + dx, py + dy, lbl, fontsize=9, fontweight="bold", color="#1E3A8A")
    ax.text(50, 83, "윗변 W1 = 220px", ha="center", fontsize=8.5, color="#475569", fontweight="bold")
    ax.text(50, 12, "아랫변 W2 = 360px (더 김! → 가로 채택)", ha="center", fontsize=8.5, color="#DC2626", fontweight="bold")
    ax.text(8, 52, "왼변\n380px", ha="center", fontsize=8, color="#475569", fontweight="bold")
    ax.text(91, 49, "오른변\n410px\n(채택!)", ha="center", fontsize=8, color="#16A34A", fontweight="bold")
    ax.text(
        50,
        49,
        "새 캔버스 크기:\nW = max(220, 360) = 360\nH = max(380, 410) = 410",
        ha="center",
        va="center",
        fontsize=8.3,
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#60A5FA"),
    )
    ax.set_xlim(0, 102)
    ax.set_ylim(2, 95)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 2: Homography Grid Un-warping Visualization
    ax = axes[1]
    ax.set_title("② 3×3 행렬(M)의 역할\n(좁은 윗칸은 늘리고 비뚤어진 칸은 폄)", fontsize=10.5, fontweight="bold", color="#2563EB", pad=8)
    ax.set_facecolor("#F8FAFC")
    src_g = np.array([[14, 72], [38, 72], [46, 22], [6, 22]], dtype=np.float32)
    dst_g = np.array([[62, 74], [94, 74], [94, 22], [62, 22]], dtype=np.float32)
    # Draw perspective grid inside left trapezoid
    for t in np.linspace(0, 1, 5):
        p_left = src_g[0] * (1 - t) + src_g[3] * t
        p_right = src_g[1] * (1 - t) + src_g[2] * t
        ax.plot([p_left[0], p_right[0]], [p_left[1], p_right[1]], color="#EF4444", lw=1.3, alpha=0.8)
        p_top = src_g[0] * (1 - t) + src_g[1] * t
        p_bot = src_g[3] * (1 - t) + src_g[2] * t
        ax.plot([p_top[0], p_bot[0]], [p_top[1], p_bot[1]], color="#EF4444", lw=1.3, alpha=0.8)
    # Draw uniform grid inside right rectangle
    for t in np.linspace(0, 1, 5):
        y = 22 + (74 - 22) * t
        x = 62 + (94 - 62) * t
        ax.plot([62, 94], [y, y], color="#16A34A", lw=1.4, alpha=0.85)
        ax.plot([x, x], [22, 74], color="#16A34A", lw=1.4, alpha=0.85)

    ax.annotate(
        "",
        xy=(60, 48),
        xytext=(46, 48),
        arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color="#1D4ED8", lw=2.5),
    )
    ax.text(53, 56, "M (3×3)", ha="center", fontsize=8.8, fontweight="bold", color="#1D4ED8")
    ax.text(26, 82, "원근 왜곡 격자\n(위가 좁고 아래가 넓음)", ha="center", fontsize=8.2, color="#991B1B", fontweight="bold")
    ax.text(78, 82, "정면 균일 격자\n(모든 칸이 똑같은 직사각형)", ha="center", fontsize=8.2, color="#14532D", fontweight="bold")
    ax.text(50, 10, "warpPerspective가 모든 픽셀을 이 공식대로 재배치!", ha="center", fontsize=8.4, color="#1E293B", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 95)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 3: Real Before -> After warp on Flow tilt_01.jpg
    ax = axes[2]
    ax.set_title("③ 실제 실사 적용 (tilt_01: 45° 마름모)\n(왼쪽 원본 → 오른쪽 반듯한 정면 복원)", fontsize=10.5, fontweight="bold", color="#16A34A", pad=8)
    scanner = DocumentScanner()
    img_t1 = load_image(PROJECT_ROOT / "data" / "flow_document_images" / "tilt_01.jpg")
    res_t1 = scanner.scan_image(img_t1, PRESETS["morph_bridge"])
    orig_rgb = cv2.cvtColor(res_t1.contour_overlay, cv2.COLOR_BGR2RGB)
    warp_rgb = cv2.cvtColor(res_t1.warped, cv2.COLOR_BGR2RGB)
    h_target = 360
    o_small = cv2.resize(orig_rgb, (int(orig_rgb.shape[1] * h_target / orig_rgb.shape[0]), h_target))
    w_small = cv2.resize(warp_rgb, (int(warp_rgb.shape[1] * h_target / warp_rgb.shape[0]), h_target))
    gap = np.full((h_target, 18, 3), 245, dtype=np.uint8)
    combined = np.hstack([o_small, gap, w_small])
    ax.imshow(combined)
    ax.set_xticks([])
    ax.set_yticks([])

    out = DIAGRAMS_DIR / "07_stage4_perspective_warp_guide.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_08_stage5_adaptive_guide() -> Path:
    """Diagram 8: Visual 3-panel beginner guide to Stage 5 (1D profile + Real Global vs Adaptive on shadow_02)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.4), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "5단계 완전 시각화: 왜 전역 이진화는 그림자에서 망하고, 적응형 이진화는 살아남는가?",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
        y=1.02,
    )

    # Panel 1: 1D brightness cross-section illustrating global vs adaptive threshold
    ax = axes[0]
    ax.set_title("① 밝기 단면 원리 비교\n(빨간 점선=전역 고정 vs 초록 실선=동네 평균)", fontsize=10.5, fontweight="bold", color="#1E293B", pad=8)
    x = np.linspace(0, 100, 400)
    # Paper background smoothly drops from 230 (bright left) to 95 (shadow right)
    bg_curve = 230.0 - 135.0 / (1.0 + np.exp(-(x - 50.0) / 6.0))
    signal = bg_curve.copy()
    # Add 4 dark text dips (2 in bright side, 2 in shadow side)
    for tx, depth in [(16, 155), (34, 150), (66, 65), (84, 60)]:
        signal -= depth * np.exp(-((x - tx) ** 2) / 3.0)

    ax.axvspan(50, 100, color="#E2E8F0", alpha=0.6, label="그림자 구역 (오른쪽 절반)")
    ax.plot(x, signal, color="#0F172A", lw=2.0, label="실제 픽셀 밝기 (골짜기=글씨)")
    ax.axhline(128, color="#DC2626", ls="--", lw=2.0, label="전역 기준선 (고정 128)")
    ax.plot(x, bg_curve - 14.0, color="#16A34A", lw=2.2, label="적응형 기준선 (동네 평균 - C)")
    ax.annotate(
        "전역(128)보다 낮아\n종이까지 전부 검정!",
        xy=(75, 95),
        xytext=(54, 165),
        fontsize=8.2,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#F87171"),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
    )
    ax.annotate(
        "초록선 아래 골짜기만\n정확히 글씨 판정!",
        xy=(66, 35),
        xytext=(12, 18),
        fontsize=8.2,
        fontweight="bold",
        color="#14532D",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FDF4", edgecolor="#4ADE80"),
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.5),
    )
    ax.set_xlabel("문서 가로 위치 (왼쪽=밝음, 오른쪽=그림자)", fontsize=9)
    ax.set_ylabel("픽셀 밝기 (0=검정, 255=흰색)", fontsize=9)
    ax.set_ylim(0, 265)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7.4, framealpha=0.94)

    # Real warped shadow_02 image for Panels 2 & 3
    scanner = DocumentScanner()
    img_s2 = load_image(PROJECT_ROOT / "data" / "flow_document_images" / "shadow_02.jpg")
    res_s2 = scanner.scan_image(img_s2, PRESETS["clahe_shadow_morph"])
    warped_gray = cv2.cvtColor(res_s2.warped, cv2.COLOR_BGR2GRAY)
    _, global_bin = cv2.threshold(warped_gray, 128, 255, cv2.THRESH_BINARY)
    adaptive_bin = res_s2.scanned

    # Panel 2: Real Global Thresholding failure
    ax = axes[1]
    ax.set_title("② 전역 이진화 (고정 기준=128)\n[실패 ×] 그림자 영역이 새까맣게 먹통!", fontsize=10.5, fontweight="bold", color="#DC2626", pad=8)
    ax.imshow(global_bin, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 3: Real Adaptive Thresholding success
    ax = axes[2]
    ax.set_title("③ 적응형 이진화 (15×15 동네 평균, C=8)\n[성공 ○] 손 그림자 사라지고 글씨·바코드 복원!", fontsize=10.5, fontweight="bold", color="#16A34A", pad=8)
    ax.imshow(adaptive_bin, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])

    out = DIAGRAMS_DIR / "08_stage5_adaptive_threshold_guide.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    configure_utf8_console()
    _setup_korean_font()
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        generate_01_architecture_pipeline(),
        generate_02_geometry_perspective_math(),
        generate_03_eda_histograms(),
        generate_04_benchmark_parameter_dilemma(),
        generate_05_real_pipeline_examples(),
        generate_06_contour_approx_guide(),
        generate_07_stage4_perspective_guide(),
        generate_08_stage5_adaptive_guide(),
    ]
    for path in outputs:
        print(f"[OK] Generated diagram: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
