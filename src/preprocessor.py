"""Image preprocessing and Canny edge detection module."""

# [1차]: 타입 어노테이션 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 미리 메모만 해두고 나중에 해석합니다.
from __future__ import annotations

# [1차]: 전처리 중간 산출물 묶음을 표현하기 위해 dataclass를 가져옵니다.
# [2차]: 흑백·블러·엣지 사진 3장을 한 세트로 묶을 상자 틀을 준비합니다.
from dataclasses import dataclass

# [1차]: 색상 변환, 가우시안 블러, Canny, CLAHE, 모폴로지 연산을 위해 cv2를 가져옵니다.
# [2차]: 사진 보정과 윤곽선 따기를 담당하는 OpenCV 미술 도구 상자를 엽니다.
import cv2

# [1차]: 이미지 배열 타입 및 차원 검사를 위해 NumPy를 가져옵니다.
# [2차]: 이미지를 숫자 행렬로 다루기 위해 NumPy를 불러옵니다.
import numpy as np

# [1차]: 스캐너 설정 객체와 홀수 커널 정규화 헬퍼를 config 모듈에서 가져옵니다.
# [2차]: 다이얼 설정표(ScanConfig)와 홀수 크기 안전 보정기를 가져옵니다.
from .config import ScanConfig, normalize_odd_kernel


# [1차]: 전처리 파이프라인의 3단계 중간 이미지(흑백, 블러, 엣지)를 담는 데이터클래스입니다.
# [2차]: "원본을 흑백으로 바꾸고 → 흐리고 → 테두리 딴" 3장의 중간 결과물을 담아두는 앨범입니다.
@dataclass
class PreprocessArtifacts:
    """Intermediate images generated during preprocessing and edge detection."""

    # [1차]: 단일 채널 흑백 변환 이미지 (H, W) uint8 배열입니다.
    # [2차]: 색상을 빼고 밝기만 남긴 흑백 사진입니다.
    grayscale: np.ndarray
    # [1차]: 가우시안 블러(및 선택적 CLAHE)가 적용된 노이즈 완화 이미지입니다.
    # [2차]: 자잘한 노이즈를 지우개로 부드럽게 문질러 둔 사진입니다.
    blurred: np.ndarray
    # [1차]: Canny 엣지 검출(및 선택적 모폴로지 닫힘) 결과 이진 엣지 맵({0, 255})입니다.
    # [2차]: 검은 도화지 위에 물체 테두리만 하얀 분필 선으로 그려낸 그림입니다.
    edges: np.ndarray


# [1차]: BGR(3채널), BGRA(4채널), 또는 이미 흑백(2차원)인 이미지를 1채널 흑백 배열로 변환합니다.
# [2차]: 컬러 사진의 빨강·초록·파랑 색종이 3장을 합쳐서 밝기(명암)만 남긴 흑백 사진 1장으로 바꿉니다.
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA image to single-channel grayscale uint8 array."""
    # [1차]: 입력 이미지가 None이거나 픽셀 수가 0이면 ValueError를 발생시킵니다.
    # [2차]: 빈 도화지가 들어오면 작업을 시작하지 않고 즉시 알립니다.
    if image is None or image.size == 0:
        # [1차]: 빈 입력 오류 메시지를 담아 예외를 발생시킵니다.
        # [2차]: "변환할 사진이 비어 있어요!"라고 안내합니다.
        raise ValueError("입력 이미지가 비어 있습니다.")
    # [1차]: 이미 2차원 (H, W) 흑백 배열이면 원본 보호를 위해 사본을 반환합니다.
    # [2차]: 이미 흑백 사진이라면 다시 바꿀 필요 없이 복사본만 건네줍니다.
    if image.ndim == 2:
        # [1차]: .copy()로 원본 배열이 외부에서 수정되지 않게 독립 복제합니다.
        # [2차]: 원본 사진에 낙서가 묻지 않게 복사본을 줍니다.
        return image.copy()
    # [1차]: 투명도(Alpha) 채널이 포함된 4채널 BGRA 이미지를 흑백으로 변환합니다.
    # [2차]: 투명 배경 PNG 사진도 당황하지 않고 깔끔하게 흑백으로 바꿉니다.
    if image.ndim == 3 and image.shape[2] == 4:
        # [1차]: OpenCV의 COLOR_BGRA2GRAY 공식을 적용합니다.
        # [2차]: 4채널 색상 정보를 밝기 1채널로 합칩니다.
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    # [1차]: 표준 3채널 BGR 컬러 이미지를 흑백으로 변환합니다.
    # [2차]: 일반 카메라 사진(BGR)을 흑백 명암 사진으로 바꿉니다.
    if image.ndim == 3 and image.shape[2] == 3:
        # [1차]: OpenCV의 가중합(0.299R + 0.587G + 0.114B)으로 흑백 변환합니다.
        # [2차]: 사람 눈이 초록빛에 가장 민감하다는 원리를 반영해 자연스러운 흑백으로 바꿉니다.
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # [1차]: 2, 3, 4채널 이외의 비정상 차원은 명확히 거부합니다.
    # [2차]: 도저히 사진이라 볼 수 없는 이상한 모양의 배열은 거절합니다.
    raise ValueError(f"지원하지 않는 이미지 차원입니다: {image.shape}")


# [1차]: 2D 가우시안 필터 컨볼루션을 적용하여 고주파 센서 노이즈와 미세 질감을 억제합니다.
# [2차]: 사진 속 자글자글한 먼지·종이결을 부드럽게 흐려서 가짜 선이 생기지 않게 다듬습니다.
def apply_gaussian_blur(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply 2D Gaussian blur to suppress high-frequency sensor and texture noise."""
    # [1차]: 입력 커널 크기를 1~51 범위의 안전한 홀수로 정규화합니다.
    # [2차]: 지우개 크기가 짝수여도 고장 나지 않도록 안전한 홀수로 맞춥니다.
    k = normalize_odd_kernel(kernel_size, minimum=1, maximum=51)
    # [1차]: 커널 크기가 1이면 블러 효과가 없으므로 연산을 생략하고 사본을 반환합니다.
    # [2차]: 지우개 크기가 1×1이면 문지르지 않은 것과 같으므로 그대로 돌려줍니다.
    if k == 1:
        # [1차]: 불필요한 컨볼루션을 건너뛰고 복사본을 반환합니다.
        # [2차]: 원본 흑백 사진 복사본을 바로 넘깁니다.
        return gray.copy()
    # [1차]: sigmaX=0으로 지정하여 커널 크기 k에 맞는 최적 가우시안 표준편차를 자동 계산합니다.
    # [2차]: 중심 픽셀에 가장 큰 비중을 두고 주변 픽셀을 종 모양 비율로 섞어 부드럽게 흐립니다.
    return cv2.GaussianBlur(gray, (k, k), 0)


# [1차]: 히스테리시스 이중 임계값을 사용하는 Canny 엣지 검출을 수행합니다.
# [2차]: 밝기가 급격히 변하는 절벽(경계선)만 골라내 검은 바탕에 흰 선으로 따냅니다.
def detect_canny_edges(blurred: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """Run Canny edge detector with hysteresis thresholding."""
    # [1차]: 하한 임계값을 0~255 정수로 안전하게 제한합니다.
    # [2차]: 약한 선 기준값을 0~255 범위 안으로 맞춥니다.
    low = max(0, min(255, int(low_threshold)))
    # [1차]: 상한 임계값을 0~255 정수로 안전하게 제한합니다.
    # [2차]: 강한 선 기준값도 0~255 범위 안으로 맞춥니다.
    high = max(0, min(255, int(high_threshold)))
    # [1차]: low > high 역전 입력 시 자동으로 순서를 바로잡습니다.
    # [2차]: 두 기준이 뒤바뀌었으면 올바른 순서로 바꿔줍니다.
    if low > high:
        # [1차]: 두 정수를 교환합니다.
        # [2차]: 작은 값이 아래(low), 큰 값이 위(high)로 가게 바꿉니다.
        low, high = high, low
    # [1차]: Sobel 기울기·비최대 억제·이중 임계값 연결을 수행해 이진 엣지 맵을 반환합니다.
    # [2차]: 두꺼운 선은 1픽셀로 얇게 깎고, 강한 선에 이어진 약한 선만 살려 깔끔한 외곽선을 만듭니다.
    return cv2.Canny(blurred, low, high)


# [1차]: 설정(ScanConfig)에 따라 흑백 → 선택적 CLAHE → 블러 → Canny → 선택적 모폴로지 닫힘을 실행합니다.
# [2차]: 원본 컬러 사진이 들어오면 밑그림(엣지)이 완성될 때까지 전처리 4단 콤보를 순서대로 돌립니다.
def preprocess_image(image: np.ndarray, config: ScanConfig) -> PreprocessArtifacts:
    """Run grayscale conversion, optional CLAHE, Gaussian blur, Canny edge, and optional closing."""
    # [1차]: 입력 이미지를 단일 채널 흑백으로 변환합니다.
    # [2차]: 먼저 색을 빼서 명암(밝기) 정보만 남깁니다.
    gray = to_grayscale(image)
    # [1차]: 원본 grayscale을 보존하기 위해 작업용 참조 변수를 분리합니다.
    # [2차]: 순수 흑백 사진은 앨범에 잘 보관하고 작업대 위로 사본을 올립니다.
    working_gray = gray

    # [1차]: 어두운 그림자 보정 옵션(use_clahe)이 켜져 있으면 국소 히스토그램 평활화를 적용합니다.
    # [2차]: 어두운 구역의 숨은 윤곽을 살려내기 위해 이미지를 8×8 조각으로 나눠 구역별 밝기를 증폭합니다.
    if config.use_clahe:
        # [1차]: 노이즈 과증폭을 막기 위해 clipLimit=2.5, 8×8 타일 그리드로 CLAHE 객체를 생성합니다.
        # [2차]: 대비를 너무 과하게 올리지 않도록 상한선(2.5)을 둔 국소 조명 보정기를 준비합니다.
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        # [1차]: 작업용 흑백 이미지에 CLAHE를 적용합니다.
        # [2차]: 어두웠던 그림자 속 대비를 확 끌어올립니다.
        working_gray = clahe.apply(working_gray)

    # [1차]: 설정된 커널 크기로 가우시안 블러를 적용해 미세 노이즈를 제거합니다.
    # [2차]: 엣지를 따기 전에 자잘한 티끌을 부드럽게 지웁니다.
    blurred = apply_gaussian_blur(working_gray, config.blur_kernel)
    # [1차]: 블러된 이미지에서 Canny 알고리즘으로 외곽선 이진 맵을 추출합니다.
    # [2차]: 밝기가 뚝 떨어지는 종이 경계선만 흰 선으로 뽑아냅니다.
    edges = detect_canny_edges(blurred, config.canny_low, config.canny_high)

    # [1차]: 모폴로지 닫힘 커널이 3 이상이면 팽창 후 침식(MORPH_CLOSE)으로 끊어진 엣지 틈을 메웁니다.
    # [2차]: 희미해서 중간에 점선처럼 끊긴 테두리를 살짝 불렸다가 다시 줄여 매끈한 실선으로 이어줍니다.
    if config.morph_kernel >= 3:
        # [1차]: 지정 크기의 사각형 구조화 요소(Structuring Element)를 생성합니다.
        # [2차]: 끊긴 선을 이어 붙일 네모난 도장(커널)을 만듭니다.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (config.morph_kernel, config.morph_kernel)
        )
        # [1차]: MORPH_CLOSE 연산으로 작은 구멍과 선 단절을 연결합니다.
        # [2차]: 끊어진 외곽선 틈새를 테이프로 메워 닫힌 도형이 되게 돕습니다.
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # [1차]: 3단계 전처리 결과물(흑백, 블러, 엣지)을 PreprocessArtifacts로 묶어 반환합니다.
    # [2차]: 완성된 3장의 중간 스케치를 앨범 상자에 담아 다음 단계로 보냅니다.
    return PreprocessArtifacts(grayscale=gray, blurred=blurred, edges=edges)
