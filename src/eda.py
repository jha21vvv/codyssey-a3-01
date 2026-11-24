"""Exploratory Data Analysis (EDA) for pixel brightness histograms and edge contrast."""

# [1차]: 타입 어노테이션 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 실행 시점까지 잠깐 보류해 두는 스위치입니다.
from __future__ import annotations

# [1차]: 통계 구조체 정의(@dataclass)와 딕셔너리 직렬화(asdict) 도구를 가져옵니다.
# [2차]: 사진 건강검진 결과표를 만드는 틀과 JSON 변환기를 꺼냅니다.
from dataclasses import asdict, dataclass

# [1차]: 메타데이터 기준 상대 경로 해석을 위해 Path를 가져옵니다.
# [2차]: 사진 파일 주소를 깔끔하게 조합하는 스마트 주소 객체입니다.
from pathlib import Path

# [1차]: 딕셔너리·리스트·범용 타입 힌트를 가져옵니다.
# [2차]: 반환 자료 형태를 미리 알려주는 라벨입니다.
from typing import Any, Dict, List

# [1차]: 히스토그램(calcHist)과 Sobel 기울기(Sobel) 계산을 위해 OpenCV를 가져옵니다.
# [2차]: 픽셀 밝기 분포와 경계선 강도를 정밀하게 재는 진단 장비입니다.
import cv2

# [1차]: 백분위수, 평균, 표준편차, 제곱근 연산을 위해 NumPy를 가져옵니다.
# [2차]: 수십만 개 픽셀의 통계를 순식간에 뽑아주는 계산기입니다.
import numpy as np

# [1차]: 기본 Canny 엣지 밀도 측정을 위해 ScanConfig를 가져옵니다.
# [2차]: 기준 다이얼 설정표를 불러옵니다.
from .config import ScanConfig

# [1차]: 한글 안전 이미지·JSON 입출력 유틸리티를 가져옵니다.
# [2차]: 다국어 택배 기사를 불러옵니다.
from .io_utils import PathLike, load_image, load_json, save_json

# [1차]: 전처리 파이프라인과 흑백 변환 함수를 가져옵니다.
# [2차]: 흑백 변환기와 Canny 엣지 밑그림 담당을 빌려옵니다.
from .preprocessor import preprocess_image, to_grayscale


# [1차]: 단일 이미지의 픽셀 밝기 분포, 대비, Sobel 기울기, 엣지 밀도를 기록하는 데이터클래스입니다.
# [2차]: 사진 한 장의 밝기·어두움·선명도를 수치로 기록한 건강검진 성적표입니다.
@dataclass
class ImageEdaStats:
    """Brightness, contrast, and edge density statistics for one image."""

    # [1차]: 데이터셋 내 이미지 고유 식별자입니다.
    # [2차]: 수험번호처럼 사진마다 붙은 고유 ID입니다.
    image_id: str
    # [1차]: 이미지 파일명입니다.
    # [2차]: 사진 파일 이름입니다.
    filename: str
    # [1차]: 4대 촬영 조건 라벨(단순 배경+정면, 그림자 있음, 45도 이상 기울어짐, 복잡한 배경)입니다.
    # [2차]: 이 사진이 어떤 시험 조건 그룹에 속하는지 적는 반 이름입니다.
    condition: str
    # [1차]: 쉬운 케이스("easy") vs 어려운 케이스("hard") 구분 라벨입니다.
    # [2차]: 쉬운 문제인지 어려운 문제인지 표시합니다.
    difficulty: str
    # [1차]: 이미지 가로 픽셀 수입니다.
    # [2차]: 사진의 가로 너비입니다.
    width: int
    # [1차]: 이미지 세로 픽셀 수입니다.
    # [2차]: 사진의 세로 높이입니다.
    height: int
    # [1차]: 전체 픽셀 밝기 산술평균(0~255)입니다.
    # [2차]: 사진 전체가 전반적으로 밝은지 어두운지 나타내는 평균 점수입니다.
    mean_brightness: float
    # [1차]: 전체 픽셀 밝기 표준편차(대비 지표)입니다.
    # [2차]: 어두운 배경과 밝은 종이가 얼마나 극명하게 차이 나는지 보여주는 대비 지수입니다.
    std_brightness: float
    # [1차]: 하위 5% 백분위 밝기값입니다.
    # [2차]: 사진에서 가장 어두운 편(하위 5%)에 속하는 밝기 기준선입니다.
    p05: float
    # [1차]: 중앙값(50% 백분위) 밝기값입니다.
    # [2차]: 모든 픽셀을 밝기순으로 세웠을 때 딱 한가운데 값입니다.
    p50: float
    # [1차]: 상위 5%(95% 백분위) 밝기값입니다.
    # [2차]: 사진에서 가장 밝은 편(상위 5%)에 속하는 밝기 기준선입니다.
    p95: float
    # [1차]: p95 - p05로 정의한 유효 동적 범위(Dynamic Range)입니다.
    # [2차]: 가장 어두운 곳부터 가장 밝은 곳까지 밝기 폭이 얼마나 넓은지 재는 수치입니다.
    dynamic_range: float
    # [1차]: 밝기 60 미만인 암부 픽셀 비율(0.0~1.0)입니다.
    # [2차]: 사진 전체에서 캄캄한 어둠(60 미만)이 차지하는 비중입니다.
    dark_pixel_ratio: float
    # [1차]: 밝기 200 초과인 고명도 픽셀 비율(0.0~1.0)입니다.
    # [2차]: 사진 전체에서 하얀 종이나 빛 반사(200 초과)가 차지하는 비중입니다.
    bright_pixel_ratio: float
    # [1차]: 전체 픽셀의 평균 Sobel 1차 미분 기울기 크기 |∇I|입니다.
    # [2차]: 사진 전체에서 밝기가 꺾이는 '경사(언덕)'가 평균적으로 얼마나 가파른지 나타냅니다.
    mean_gradient: float
    # [1차]: 기본 Canny 결과에서 흰색 엣지 픽셀(>0)이 차지하는 비율입니다.
    # [2차]: 밑그림 전체 넓이 중 하얀 테두리 선이 몇 %나 그려졌는지 나타냅니다.
    edge_density: float


# [1차]: 1채널 흑백 이미지의 0~255 밝기 구간별 정규화 히스토그램(합=1.0)을 계산합니다.
# [2차]: 0번(검정)부터 255번(흰색)까지 각 밝기 방에 픽셀이 전체의 몇 %씩 들어 있는지 표로 만듭니다.
def compute_histogram(gray: np.ndarray) -> np.ndarray:
    """Compute normalized 256-bin intensity histogram for a grayscale image."""
    # [1차]: OpenCV calcHist로 256개 빈(0~255)의 픽셀 개수를 집계해 1차원 벡터로 펼칩니다.
    # [2차]: 0~255 밝기별로 픽셀이 몇 개씩 있는지 빠르게 셉니다.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    # [1차]: 전체 픽셀 총합을 구해 이미지 해상도와 무관하게 비교할 수 있도록 정규화합니다.
    # [2차]: 사진 크기가 달라도 공평하게 비교하도록 전체 개수로 나눠 비율(합계 1.0)로 바꿉니다.
    total = float(hist.sum())
    # [1차]: 총합이 0보다 크면 모든 빈을 total로 나눕니다.
    # [2차]: 0으로 나누는 오류를 피하면서 각 밝기의 비중을 계산합니다.
    if total > 0:
        hist = hist / total
    # [1차]: 정밀도를 유지한 float64 1차원 배열(길이 256)을 반환합니다.
    # [2차]: 256칸짜리 깔끔한 밝기 분포 그래프 데이터를 돌려줍니다.
    return hist.astype(np.float64)


# [1차]: 단일 이미지의 히스토그램, 밝기 분위수, Sobel 기울기, Canny 엣지 밀도를 종합 산출합니다.
# [2차]: 사진 한 장을 건강검진 기계에 넣어 밝기·대비·경사도·선 밀도를 한 번에 측정합니다.
def analyze_image_brightness(
    image: np.ndarray,
    image_id: str,
    filename: str,
    condition: str,
    difficulty: str,
    config: ScanConfig | None = None,
) -> tuple[ImageEdaStats, np.ndarray]:
    """Compute pixel distribution, Sobel gradient strength, and Canny edge density."""
    # [1차]: 설정이 없으면 기본 ScanConfig를 사용합니다.
    # [2차]: 표준 검사 기준표를 준비합니다.
    cfg = config or ScanConfig()
    # [1차]: 컬러 이미지를 흑백 1채널로 변환합니다.
    # [2차]: 색을 빼고 밝기 지도만 남깁니다.
    gray = to_grayscale(image)
    # [1차]: 256-bin 정규화 밝기 히스토그램을 계산합니다.
    # [2차]: 밝기 분포 그래프 데이터를 뽑습니다.
    hist = compute_histogram(gray)

    # [1차]: 5%, 50%, 95% 백분위수를 한 번의 NumPy 호출로 계산합니다.
    # [2차]: 어두운 5% 지점, 정중앙 50% 지점, 밝은 95% 지점을 한꺼번에 잽니다.
    p05, p50, p95 = np.percentile(gray, [5, 50, 95])
    # [1차]: 3×3 Sobel 필터로 가로 방향(x) 밝기 미분 기울기를 계산합니다.
    # [2차]: 왼쪽에서 오른쪽으로 갈 때 밝기가 얼마나 급격히 변하는지 잽니다.
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # [1차]: 3×3 Sobel 필터로 세로 방향(y) 밝기 미분 기울기를 계산합니다.
    # [2차]: 위에서 아래로 갈 때 밝기가 얼마나 급격히 변하는지 잽니다.
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # [1차]: 유클리드 크기 sqrt(gx^2 + gy^2)로 각 픽셀의 종합 경계 기울기 크기를 구합니다.
    # [2차]: 가로·세로 경사를 피타고라스 정리로 합쳐 "이 픽셀의 절벽 가파름"을 계산합니다.
    grad_mag = np.sqrt(gx * gx + gy * gy)

    # [1차]: 기본 전처리 파이프라인을 돌려 Canny 엣지 밀도(흰 픽셀 비율)를 측정합니다.
    # [2차]: 실제로 밑그림을 따봤을 때 화면의 몇 %가 하얀 선으로 채워지는지 잽니다.
    prep = preprocess_image(image, cfg)
    edge_density = float(np.mean(prep.edges > 0))

    # [1차]: 모든 지표를 소수점 정리해 ImageEdaStats 데이터클래스로 패키징합니다.
    # [2차]: 측정한 모든 건강 수치를 깔끔하게 반올림해 결과표 한 장으로 묶습니다.
    stats = ImageEdaStats(
        image_id=image_id,
        filename=filename,
        condition=condition,
        difficulty=difficulty,
        width=int(gray.shape[1]),
        height=int(gray.shape[0]),
        mean_brightness=round(float(np.mean(gray)), 2),
        std_brightness=round(float(np.std(gray)), 2),
        p05=round(float(p05), 2),
        p50=round(float(p50), 2),
        p95=round(float(p95), 2),
        dynamic_range=round(float(p95 - p05), 2),
        dark_pixel_ratio=round(float(np.mean(gray < 60)), 4),
        bright_pixel_ratio=round(float(np.mean(gray > 200)), 4),
        mean_gradient=round(float(np.mean(grad_mag)), 2),
        edge_density=round(edge_density, 4),
    )
    # [1차]: 통계 객체와 256-bin 히스토그램 배열을 튜플로 반환합니다.
    # [2차]: 건강검진 성적표와 밝기 그래프 데이터를 함께 돌려줍니다.
    return stats, hist


# [1차]: metadata.json에 등록된 전체 20장 이미지의 EDA를 수행하고 조건별·난이도별 평균을 집계합니다.
# [2차]: 20장 시험 데이터 전체를 단체 건강검진하고, '쉬운 반 vs 어려운 반' 평균 차트를 작성합니다.
def run_dataset_eda(
    metadata_path: PathLike,
    output_json: PathLike | None = None,
) -> Dict[str, Any]:
    """Run histogram and brightness distribution analysis across all dataset images."""
    # [1차]: 메타데이터 파일 경로를 해석하고 JSON 내용을 불러옵니다.
    # [2차]: 20장 사진 명부(metadata.json)를 펼쳐 읽습니다.
    meta_file = Path(metadata_path)
    meta = load_json(meta_file)
    base_dir = meta_file.parent

    # [1차]: 이미지별 기록과 조건별·난이도별 히스토그램 누적 바구니를 초기화합니다.
    # [2차]: 반별 평균 그래프를 그리기 위해 성적표 모음 바구니를 준비합니다.
    records: List[Dict[str, Any]] = []
    cond_hists: Dict[str, List[np.ndarray]] = {}
    diff_hists: Dict[str, List[np.ndarray]] = {"easy": [], "hard": []}

    # [1차]: 명부의 20장 이미지를 순회하며 개별 밝기·기울기 분석을 수행합니다.
    # [2차]: 20장 사진을 한 장씩 불러와 차례대로 검진합니다.
    for item in meta["images"]:
        img_path = base_dir / item["relative_path"]
        image = load_image(img_path)
        stats, hist = analyze_image_brightness(
            image=image,
            image_id=item["id"],
            filename=item["filename"],
            condition=item["condition"],
            difficulty=item["difficulty"],
        )
        # [1차]: 개별 결과를 딕셔너리로 누적하고 해당 조건·난이도 그룹에 히스토그램을 추가합니다.
        # [2차]: 검진표를 전체 명부에 적고, 소속 반(조건/난이도) 바구니에도 분포 그래프를 넣습니다.
        records.append(asdict(stats))
        cond_hists.setdefault(item["condition"], []).append(hist)
        diff_hists.setdefault(item["difficulty"], []).append(hist)

    # [1차]: 4개 촬영 조건별 평균 밝기·표준편차·동적범위·기울기·평균 히스토그램을 집계합니다.
    # [2차]: '단순+정면', '그림자', '45° 기울기', '복잡한 배경' 4개 반의 평균 성적표를 만듭니다.
    condition_summary: Dict[str, Dict[str, float]] = {}
    for cond, hists in cond_hists.items():
        subset = [r for r in records if r["condition"] == cond]
        condition_summary[cond] = {
            "count": len(subset),
            "mean_brightness": round(float(np.mean([r["mean_brightness"] for r in subset])), 2),
            "std_brightness": round(float(np.mean([r["std_brightness"] for r in subset])), 2),
            "dynamic_range": round(float(np.mean([r["dynamic_range"] for r in subset])), 2),
            "dark_pixel_ratio": round(float(np.mean([r["dark_pixel_ratio"] for r in subset])), 4),
            "mean_gradient": round(float(np.mean([r["mean_gradient"] for r in subset])), 2),
            "edge_density": round(float(np.mean([r["edge_density"] for r in subset])), 4),
            "mean_histogram": [round(float(v), 6) for v in np.mean(hists, axis=0)],
        }

    # [1차]: 쉬운 케이스(5장) vs 어려운 케이스(15장) 평균 분포 및 통계를 집계합니다.
    # [2차]: 쉬운 반과 어려운 반의 평균 차이를 한눈에 보도록 요약합니다.
    difficulty_summary: Dict[str, Dict[str, Any]] = {}
    for diff, hists in diff_hists.items():
        if not hists:
            continue
        subset = [r for r in records if r["difficulty"] == diff]
        difficulty_summary[diff] = {
            "count": len(subset),
            "mean_brightness": round(float(np.mean([r["mean_brightness"] for r in subset])), 2),
            "std_brightness": round(float(np.mean([r["std_brightness"] for r in subset])), 2),
            "dark_pixel_ratio": round(float(np.mean([r["dark_pixel_ratio"] for r in subset])), 4),
            "mean_gradient": round(float(np.mean([r["mean_gradient"] for r in subset])), 2),
            "edge_density": round(float(np.mean([r["edge_density"] for r in subset])), 4),
            "mean_histogram": [round(float(v), 6) for v in np.mean(hists, axis=0)],
        }

    # [1차]: 전체 EDA 요약을 구조화하고 요청 시 UTF-8 JSON 파일로 저장한 뒤 반환합니다.
    # [2차]: 완성된 종합 진단 보고서를 JSON 파일로 저장하고 결과 딕셔너리를 돌려줍니다.
    summary: Dict[str, Any] = {
        "total_images": len(records),
        "condition_summary": condition_summary,
        "difficulty_summary": difficulty_summary,
        "images": records,
    }
    if output_json is not None:
        save_json(output_json, summary)
    return summary
