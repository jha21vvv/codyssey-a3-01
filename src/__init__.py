"""OpenCV + NumPy Document Scanner package exports."""

# [1차]: 설정 데이터클래스와 실험 프리셋, 홀수 커널 정규화 함수를 config 모듈에서 가져옵니다.
# [2차]: 요리 레시피(ScanConfig)와 추천 세트 메뉴(PRESETS)를 매장 입구 메뉴판에 올려놓습니다.
from .config import PRESETS, ScanConfig, normalize_odd_kernel

# [1차]: 윤곽선 검출 결과 클래스와 문서 사각형 탐색 함수를 contour_detector 모듈에서 가져옵니다.
# [2차]: 스케치북에서 네모 종이를 찾아내는 가위 담당 직원을 호출합니다.
from .contour_detector import ContourDetectionResult, find_document_contour

# [1차]: EDA 통계 구조체와 밝기 히스토그램 분석 함수들을 eda 모듈에서 가져옵니다.
# [2차]: 사진의 밝기 상태를 진단하는 건강검진 센터 기능을 입구로 연결합니다.
from .eda import ImageEdaStats, analyze_image_brightness, compute_histogram, run_dataset_eda

# [1차]: 20장 정량 평가 및 파라미터 스윕 함수들을 evaluator 모듈에서 가져옵니다.
# [2차]: 20문제 시험지를 자동 채점하고 성적표를 뽑는 채점관을 불러옵니다.
from .evaluator import CONDITION_ORDER, SingleEvalRecord, evaluate_dataset, run_parameter_sweep

# [1차]: 수동 4점 좌표 정렬, 투시 변환, IoU 및 꼭짓점 오차 계산 함수를 geometry 모듈에서 가져옵니다.
# [2차]: 비스듬한 종이를 네 모서리에서 잡아당겨 반듯하게 다림질하는 기하학 장인을 모셔옵니다.
from .geometry import (
    compute_corner_error,
    compute_output_dimensions,
    compute_quad_iou,
    four_point_transform,
    order_points,
    order_points_naive_sum_diff,
)

# [1차]: 실시간 트랙바 GUI 및 웹캠 스캔 실행 함수를 gui 모듈에서 가져옵니다.
# [2차]: 마우스로 다이얼을 돌리거나 웹캠으로 찍는 현장 촬영 스튜디오 기능을 연결합니다.
from .gui import run_interactive_gui, run_webcam_mode

# [1차]: UTF-8 콘솔 설정 및 한글 경로 안전 이미지·JSON 입출력 함수를 io_utils 모듈에서 가져옵니다.
# [2차]: 한글 파일 주소도 절대 잃어버리지 않는 다국어 택배 기사를 배치합니다.
from .io_utils import configure_utf8_console, load_image, load_json, save_image, save_json

# [1차]: 전체 5단계 스캐너 오케스트레이터와 몽타주 생성·저장 함수를 pipeline 모듈에서 가져옵니다.
# [2차]: 모든 공정을 한 줄로 엮어 돌리는 공장 컨베이어 벨트 지휘자를 연결합니다.
from .pipeline import DocumentScanner, ScanResult, build_stage_montage, save_stage_outputs

# [1차]: 적응형 이진화 스캔 후처리 함수들을 postprocessor 모듈에서 가져옵니다.
# [2차]: 그림자를 걷어내고 글씨만 또렷하게 인쇄하는 흑백 복사기 담당을 불러옵니다.
from .postprocessor import apply_adaptive_threshold, postprocess_scan

# [1차]: 흑백 변환, 가우시안 블러, Canny 엣지 검출 함수들을 preprocessor 모듈에서 가져옵니다.
# [2차]: 사진의 먼지를 털고 윤곽 밑그림을 그려주는 스케치 담당을 연결합니다.
from .preprocessor import (
    PreprocessArtifacts,
    apply_gaussian_blur,
    detect_canny_edges,
    preprocess_image,
    to_grayscale,
)

# [1차]: `from src import *` 사용 시 외부에 공개할 심볼 목록을 명시적으로 선언합니다.
# [2차]: 백화점 안내판에 "우리 매장 공식 취급 품목" 목록을 깔끔하게 적어둡니다.
__all__ = [
    "CONDITION_ORDER",
    "ContourDetectionResult",
    "DocumentScanner",
    "ImageEdaStats",
    "PRESETS",
    "PreprocessArtifacts",
    "ScanConfig",
    "ScanResult",
    "SingleEvalRecord",
    "analyze_image_brightness",
    "apply_adaptive_threshold",
    "apply_gaussian_blur",
    "build_stage_montage",
    "compute_corner_error",
    "compute_histogram",
    "compute_output_dimensions",
    "compute_quad_iou",
    "configure_utf8_console",
    "detect_canny_edges",
    "evaluate_dataset",
    "find_document_contour",
    "four_point_transform",
    "load_image",
    "load_json",
    "normalize_odd_kernel",
    "order_points",
    "order_points_naive_sum_diff",
    "postprocess_scan",
    "preprocess_image",
    "run_dataset_eda",
    "run_interactive_gui",
    "run_parameter_sweep",
    "run_webcam_mode",
    "save_image",
    "save_json",
    "save_stage_outputs",
    "to_grayscale",
]
