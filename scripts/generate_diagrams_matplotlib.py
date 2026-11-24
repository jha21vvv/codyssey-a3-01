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
    """Diagram 3: Real 20-image Brightness Histogram Comparison & Dark-Image Edge Analysis."""
    eda = json.loads((PROJECT_ROOT / "outputs" / "eda_summary.json").read_text(encoding="utf-8"))
    bins = np.arange(256)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.6), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "20장 테스트 데이터셋 이미지 탐색적 분석(EDA) — 밝기 히스토그램 및 조건별 대비/기울기 특성",
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
    for cond, (color, lw, ls) in palette.items():
        hist = np.array(eda["condition_summary"][cond]["mean_histogram"])
        smooth = np.convolve(hist, np.ones(5) / 5.0, mode="same")
        ax.plot(bins, smooth, color=color, linewidth=lw, linestyle=ls, label=f"{cond} (n=5)")

    ax.annotate(
        "쉬운 케이스: 배경(45)과 종이(244)의\n뚜렷한 양봉(Bimodal) 분리 → 엣지 선명",
        xy=(242, 0.045),
        xytext=(108, 0.049),
        fontsize=8.6,
        fontweight="bold",
        color="#14532D",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor="#4ADE80"),
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.6),
    )
    ax.annotate(
        "그림자 케이스: 암부(0~50)로 픽셀 쏠림\n→ 밝기차(ΔI) 축소로 Canny 미검출",
        xy=(22, 0.024),
        xytext=(62, 0.032),
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
    ax.set_ylim(0, 0.060)
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
    eval_def = json.loads((PROJECT_ROOT / "outputs" / "eval_summary.json").read_text(encoding="utf-8"))

    preset_names = ["default", "sensitive_shadow", "strong_denoise", "strict_edges", "morph_bridge"]
    preset_labels = [
        "default\n(k5, 50/150)",
        "sensitive\n(k3, 20/70+CLAHE)",
        "strong_denoise\n(k9, 80/200+M3)",
        "strict_edges\n(k7, 100/220)",
        "morph_bridge\n(k5, 40/120+M5)",
    ]
    image_ids = list(sweep["per_image_matrix"].keys())

    matrix = np.zeros((len(image_ids), len(preset_names)), dtype=np.int32)
    for r, img_id in enumerate(image_ids):
        for c, pname in enumerate(preset_names):
            matrix[r, c] = 1 if sweep["per_image_matrix"][img_id][pname] else 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6), dpi=180, gridspec_kw={"width_ratios": [1.22, 1.0]})
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "파라미터 딜레마(Parameter Dilemma) 정량 실증 — 20장 × 5개 프리셋 성공/실패 매트릭스 및 조건별 성공률",
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
    for special_id, note_color in [("shadow_04", "#2563EB"), ("tilt_04", "#7C3AED"), ("complex_02", "#D97706")]:
        r_idx = image_ids.index(special_id)
        rect = patches.Rectangle((-0.48, r_idx - 0.48), len(preset_names) - 0.04, 0.96, fill=False, edgecolor=note_color, lw=2.3)
        ax.add_patch(rect)

    for boundary in (4.5, 9.5, 14.5):
        ax.axhline(boundary, color="#334155", linewidth=1.8)

    ax.set_xticks(np.arange(len(preset_names)))
    ax.set_xticklabels(preset_labels, fontsize=8.5)
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

    # Oracle per condition
    oracle_rates = []
    for c_idx, cond in enumerate(conds):
        ids_in_cond = image_ids[c_idx * 5 : (c_idx + 1) * 5]
        ok = sum(1 for i in ids_in_cond if any(sweep["per_image_matrix"][i].values()))
        oracle_rates.append(ok / 5.0 * 100.0)
    oracle_rates.append(sweep["oracle_union_success"] / 20.0 * 100.0)

    x = np.arange(len(short_conds))
    w = 0.2
    series = [
        ("default (45%)", _rates("default"), "#64748B"),
        ("strong_denoise (65%)", _rates("strong_denoise"), "#2563EB"),
        ("morph_bridge (65%)", _rates("morph_bridge"), "#10B981"),
        ("Oracle 합집합 (75%)", oracle_rates, "#F59E0B"),
    ]
    for i, (lbl, vals, col) in enumerate(series):
        bars = ax2.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=col, label=lbl)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=6.7)

    ax2.set_title("② 조건별 성공률 비교 — 단일 파라미터 한계(65%) vs Oracle(75%)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_conds, fontsize=9.2)
    ax2.set_ylabel("검출 성공률 (%)", fontsize=10)
    ax2.set_ylim(0, 118)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.5)

    ax2.text(
        2.0,
        -22,
        "핵심 결론: strong_denoise(complex_02 성공) ↔ morph_bridge(shadow_04 성공) ↔ sensitive(tilt_04 성공)\n"
        "서로 다른 이미지가 정반대 파라미터를 요구하며, 5개 이미지(25%)는 어떤 규칙 조합으로도 해결 불가!",
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
    """Diagram 5: Real visual side-by-side results (Success, 46° Tilt, Shadow Rescue, Occlusion Fail)."""
    scanner = DocumentScanner()
    cases = [
        ("easy_01.jpg", PRESETS["default"], "① easy_01\n(단순+정면 / Default)"),
        ("tilt_01.jpg", PRESETS["default"], "② tilt_01\n(46° 마름모 / 편각 정렬)"),
        ("shadow_04.jpg", PRESETS["morph_bridge"], "③ shadow_04\n(morph_bridge 구조)"),
        ("complex_05.jpg", PRESETS["default"], "④ complex_05\n(모서리 가림 실패)"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(15.0, 12.2), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "실제 20장 벤치마크 파이프라인 실행 결과 시각화 (원본 → Canny 엣지 → 윤곽선 검출 → 스캔 결과)",
        fontsize=14,
        fontweight="bold",
        color="#0F172A",
    )

    col_titles = ["1. 입력 원본", "2. Canny 엣지", "3. 윤곽선/꼭짓점 오버레이", "4. 최종 적응형 이진화 스캔"]

    for row_idx, (fname, cfg, row_label) in enumerate(cases):
        img = load_image(PROJECT_ROOT / "data" / "test_images" / fname)
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
    ]
    for path in outputs:
        print(f"[OK] Generated diagram: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
