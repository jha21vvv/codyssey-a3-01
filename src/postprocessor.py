"""Post-processing module for clean scanned document rendering via adaptive thresholding."""

# [1차]: 타입 어노테이션 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 나중에 해석하도록 미리 켜둡니다.
from __future__ import annotations

# [1차]: 적응형 이진화(cv2.adaptiveThreshold) 수행을 위해 OpenCV를 가져옵니다.
# [2차]: 펴진 문서를 깔끔한 흑백 스캔본으로 인쇄할 복사기 엔진(OpenCV)을 꺼냅니다.
import cv2

# [1차]: 이미지 배열 타입 힌트를 위해 NumPy를 가져옵니다.
# [2차]: 이미지를 숫자 행렬로 다루기 위해 NumPy를 불러옵니다.
import numpy as np

# [1차]: 스캐너 설정 객체와 홀수 블록 크기 보정 함수를 config 모듈에서 가져옵니다.
# [2차]: 복사기 설정표(ScanConfig)와 홀수 크기 안전 보정기를 가져옵니다.
from .config import ScanConfig, normalize_odd_kernel

# [1차]: 입력 워핑 이미지를 단일 채널 흑백으로 변환하기 위해 to_grayscale 함수를 가져옵니다.
# [2차]: 이진화하기 전에 먼저 색을 빼주는 흑백 변환기를 빌려옵니다.
from .preprocessor import to_grayscale


# [1차]: 가우시안 가중 지역 평균 기반 적응형 이진화를 적용해 선명한 흑백 스캔 이미지를 생성합니다.
# [2차]: 화면 전체에 똑같은 잣대를 대지 않고 동네(block_size)마다 밝기 평균을 따로 재어 그림자 속 글씨도 살려냅니다.
def apply_adaptive_threshold(
    warped_image: np.ndarray,
    block_size: int = 15,
    c_value: int = 8,
) -> np.ndarray:
    """Convert warped document into crisp black-and-white scan using Gaussian adaptive threshold."""
    # [1차]: 컬러 또는 흑백 워핑 이미지를 1채널 uint8 흑백 배열로 통일합니다.
    # [2차]: 먼저 컬러 종이 사진에서 색을 빼고 밝기 지도만 남깁니다.
    gray = to_grayscale(warped_image)
    # [1차]: OpenCV adaptiveThreshold 요구 규격(3 이상 홀수)에 맞게 블록 크기를 정규화합니다.
    # [2차]: 동네 크기가 짝수거나 너무 작으면 고장 나므로 안전한 홀수 창문 크기로 맞춥니다.
    safe_block = normalize_odd_kernel(block_size, minimum=3, maximum=99)
    # [1차]: 각 픽셀 주변 safe_block×safe_block 가우시안 가중 평균에서 c_value를 뺀 값을 임계값으로 이진화합니다.
    # [2차]: "우리 동네 평균 밝기보다 c_value 이상 어두우면 글씨(검정 0), 아니면 깨끗한 종이(흰색 255)!"로 인쇄합니다.
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        safe_block,
        int(c_value),
    )


# [1차]: ScanConfig 설정값을 읽어 적응형 이진화 후처리를 실행하는 래퍼 함수입니다.
# [2차]: 설정 카드에 적힌 동네 크기와 깐깐함 수치(C) 그대로 흑백 스캔 버튼을 눌러줍니다.
def postprocess_scan(warped_image: np.ndarray, config: ScanConfig) -> np.ndarray:
    """Apply configured adaptive thresholding post-processing to a warped document image."""
    # [1차]: config의 adaptive_block_size와 adaptive_c를 전달해 최종 이진 스캔본을 반환합니다.
    # [2차]: 설정값대로 복사기를 돌려 완성된 흑백 스캔 문서를 건네줍니다.
    return apply_adaptive_threshold(
        warped_image,
        block_size=config.adaptive_block_size,
        c_value=config.adaptive_c,
    )
