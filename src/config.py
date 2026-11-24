"""Configuration and parameter normalization for the OpenCV Document Scanner."""

# [1차]: 타입 어노테이션의 지연 평가를 활성화합니다.
# [2차]: 파이썬이 타입 이름표를 실행 시점까지 잠깐 보류해 두도록 설정합니다.
from __future__ import annotations

# [1차]: 설정 객체 정의(@dataclass)와 딕셔너리 변환(asdict) 도구를 가져옵니다.
# [2차]: 설정값 서랍장을 빠르게 찍어내는 틀과 JSON용 장부 변환기를 꺼냅니다.
from dataclasses import asdict, dataclass

# [1차]: 딕셔너리 타입 힌트(Dict)를 가져옵니다.
# [2차]: 이름표(Key)와 값(Value)을 묶는 사전 형태임을 표시하는 라벨입니다.
from typing import Dict


# [1차]: 커널 크기를 [minimum, maximum] 범위로 제한하고 반드시 홀수 정수로 보정하는 함수입니다.
# [2차]: 가우시안 필터처럼 한가운데 중심 픽셀이 꼭 필요한 도장을 위해 짝수 크기를 안전한 홀수로 맞춰줍니다.
def normalize_odd_kernel(value: int, minimum: int = 1, maximum: int = 51) -> int:
    """Clamp kernel size to [minimum, maximum] and ensure it is an odd integer."""
    # [1차]: 입력값을 정수로 바꾼 뒤 minimum 이상, maximum 이하 범위로 클램핑합니다.
    # [2차]: 에어컨 온도 조절기처럼 최저·최고 한계선 밖으로 넘어가지 않게 울타리 안에 가둡니다.
    clamped = max(minimum, min(maximum, int(value)))
    # [1차]: 2로 나눈 나머지가 0이면 짝수이므로 홀수로 1칸 이동시킵니다.
    # [2차]: 좌우 대칭 중심을 만들기 위해 짝수 숫자가 오면 한 칸 올리거나 내려 홀수로 바꿉니다.
    if clamped % 2 == 0:
        # [1차]: 상한선 미만이면 1을 더하고, 이미 상한선이면 1을 빼서 범위를 유지합니다.
        # [2차]: 천장에 닿지 않았으면 한 칸 위로, 천장에 닿았으면 한 칸 아래로 비켜 섭니다.
        clamped = clamped + 1 if clamped < maximum else clamped - 1
    # [1차]: 어떤 경우에도 1 이상의 양의 홀수를 보장하여 반환합니다.
    # [2차]: 아무리 작은 값이 들어와도 최소 1×1 크기는 지켜서 돌려줍니다.
    return max(1, clamped)


# [1차]: 스캐너의 전처리·윤곽선·후처리 모든 하이퍼파라미터를 담는 데이터클래스입니다.
# [2차]: 문서 스캐너 기계의 모든 다이얼 눈금을 한 장의 설정 카드에 정리한 것입니다.
@dataclass
class ScanConfig:
    """Parameters controlling preprocessing, contour detection, and postprocessing."""

    # [1차]: 가우시안 블러 커널 크기(홀수 정수, 기본값 5).
    # [2차]: 사진 노이즈를 문지를 지우개의 크기(5×5 픽셀)입니다.
    blur_kernel: int = 5
    # [1차]: Canny 엣지 검출의 하한 임계값(약한 엣지 연결 기준, 기본값 50).
    # [2차]: "이 정도 밝기 변화는 강한 선 옆에 붙어 있으면 인정해 줄게" 하는 최소 커트라인입니다.
    canny_low: int = 50
    # [1차]: Canny 엣지 검출의 상한 임계값(확실한 엣지 판정 기준, 기본값 150).
    # [2차]: "이보다 밝기 변화가 크면 무조건 확실한 테두리 선이야!"라고 도장 찍는 기준입니다.
    canny_high: int = 150
    # [1차]: 모폴로지 닫힘 연산 커널 크기(0이면 미적용, 3 이상 홀수면 적용).
    # [2차]: 끊어진 점선 테두리를 이어 붙이는 테이프의 폭(0이면 테이프를 쓰지 않음)입니다.
    morph_kernel: int = 0
    # [1차]: 전체 이미지 면적 대비 유효 문서 윤곽선의 최소 비율(기본값 5%).
    # [2차]: 화면의 5%도 안 되는 작은 먼지나 글자 조각은 문서 후보에서 탈락시킵니다.
    min_area_ratio: float = 0.05
    # [1차]: 전체 이미지 면적 대비 유효 문서 윤곽선의 최대 비율(기본값 98%).
    # [2차]: 화면 테두리 전체(100%)를 문서로 착각하는 오류를 막기 위한 상한선입니다.
    max_area_ratio: float = 0.98
    # [1차]: approxPolyDP 다각형 근사 허용 오차 비율(둘레 대비 기본 2%).
    # [2차]: 울퉁불퉁한 선을 직선으로 펼 때 둘레의 2% 정도 오차는 눈감아 주는 여유분입니다.
    approx_epsilon_ratio: float = 0.02
    # [1차]: 적응형 이진화 시 지역 평균을 계산할 이웃 블록 크기(홀수, 기본 15).
    # [2차]: 글씨인지 배경인지 판단할 때 살펴보는 '우리 동네' 크기(15×15 픽셀)입니다.
    adaptive_block_size: int = 15
    # [1차]: 적응형 이진화에서 지역 평균에서 빼줄 상수 C(기본 8).
    # [2차]: 동네 평균 밝기보다 최소 8만큼은 더 어두워야 진짜 글씨(검정)로 인정하는 깐깐함 수치입니다.
    adaptive_c: int = 8
    # [1차]: 전처리 단계에서 CLAHE 국소 대비 보정을 적용할지 여부(기본 False).
    # [2차]: 어두운 방에서 손전등으로 구역마다 밝기를 억지로 끌어올릴지 결정하는 스위치입니다.
    use_clahe: bool = False
    # [1차]: 검출된 4각형 윤곽선이 볼록(Convex) 도형이어야만 채택할지 여부(기본 True).
    # [2차]: 안쪽으로 움푹 파인 별 모양·L자 모양은 정상 종이가 아니므로 걸러내는 검사입니다.
    require_convex: bool = True

    # [1차]: 데이터클래스 인스턴스 생성 직후 모든 필드를 안전한 값으로 자동 검증·정규화합니다.
    # [2차]: 공장 출고 직전에 모든 다이얼이 안전한 범위 안에 있는지 최종 검수하는 단계입니다.
    def __post_init__(self) -> None:
        # [1차]: blur_kernel을 1~51 범위의 홀수로 보정합니다.
        # [2차]: 짝수 블러 크기가 들어와도 OpenCV가 에러 내지 않게 홀수로 다듬습니다.
        self.blur_kernel = normalize_odd_kernel(self.blur_kernel, minimum=1, maximum=51)
        # [1차]: canny_low를 0~255 정수로 클램핑합니다.
        # [2차]: 밝기 범위(0~255)를 벗어난 하한 임계값을 정상 범위로 되돌립니다.
        self.canny_low = max(0, min(255, int(self.canny_low)))
        # [1차]: canny_high를 0~255 정수로 클램핑합니다.
        # [2차]: 상한 임계값도 0~255 사이로 안전하게 고정합니다.
        self.canny_high = max(0, min(255, int(self.canny_high)))
        # [1차]: 하한이 상한보다 크면 두 값을 맞바꾸어 역전 오류를 방지합니다.
        # [2차]: 최저 온도가 최고 온도보다 높게 입력되면 두 숫자의 자리를 바르게 바꿔줍니다.
        if self.canny_low > self.canny_high:
            # [1차]: 파이썬 튜플 언패킹으로 두 변수의 값을 임시 변수 없이 교환합니다.
            # [2차]: 왼손과 오른손에 든 카드를 한 번에 맞바꿉니다.
            self.canny_low, self.canny_high = self.canny_high, self.canny_low
        # [1차]: 하한과 상한이 완전히 같으면 상한을 1 올려 히스테리시스 구간을 확보합니다.
        # [2차]: 두 기준선이 똑같으면 구별이 안 되므로 최소 1칸의 간격을 벌려줍니다.
        if self.canny_high == self.canny_low:
            # [1차]: 255를 넘지 않도록 상한값을 canny_low + 1로 설정합니다.
            # [2차]: 최대치(255)를 넘지 않는 선에서 상한선을 한 눈금 올립니다.
            self.canny_high = min(255, self.canny_low + 1)

        # [1차]: morph_kernel이 0 이하이면 모폴로지 연산 비활성화(0)로 설정합니다.
        # [2차]: 테이프 폭이 0 이하이면 테이프를 아예 붙이지 않기로 표시합니다.
        if int(self.morph_kernel) <= 0:
            # [1차]: 비활성화 상태값 0을 저장합니다.
            # [2차]: 스위치를 OFF(0)로 고정합니다.
            self.morph_kernel = 0
        else:
            # [1차]: 양수인 경우 3~21 범위의 홀수로 정규화합니다.
            # [2차]: 테이프를 쓸 때는 최소 3×3 이상의 홀수 크기만 허용합니다.
            self.morph_kernel = normalize_odd_kernel(self.morph_kernel, minimum=3, maximum=21)

        # [1차]: 최소 면적 비율을 0.1%~90% 사이 실수로 제한합니다.
        # [2차]: 너무 작거나 너무 큰 면적 기준이 들어오지 않게 안전 구간에 맞춥니다.
        self.min_area_ratio = max(0.001, min(0.9, float(self.min_area_ratio)))
        # [1차]: 최대 면적 비율은 항상 최소 면적 비율보다 최소 1%p 이상 크게 유지합니다.
        # [2차]: 상한선이 하한선보다 작아지는 모순을 막기 위해 최소 1% 여유를 둡니다.
        self.max_area_ratio = max(self.min_area_ratio + 0.01, min(1.0, float(self.max_area_ratio)))
        # [1차]: 다각형 근사 오차 비율을 0.5%~15% 범위로 제한합니다.
        # [2차]: 직선 펴기 허용 오차가 너무 빡빡하거나 너무 느슨하지 않게 조절합니다.
        self.approx_epsilon_ratio = max(0.005, min(0.15, float(self.approx_epsilon_ratio)))
        # [1차]: 적응형 이진화 블록 크기를 3~99 범위의 홀수로 보정합니다.
        # [2차]: 동네 평균을 낼 창문 크기를 OpenCV가 요구하는 3 이상 홀수로 맞춥니다.
        self.adaptive_block_size = normalize_odd_kernel(self.adaptive_block_size, minimum=3, maximum=99)
        # [1차]: 상수 C를 정수로 변환합니다.
        # [2차]: 밝기 보정 상수를 깔끔한 정수 형태로 저장합니다.
        self.adaptive_c = int(self.adaptive_c)
        # [1차]: CLAHE 사용 여부를 명시적 불리언(bool)으로 변환합니다.
        # [2차]: 스위치 값을 확실한 참/거짓(True/False)으로 통일합니다.
        self.use_clahe = bool(self.use_clahe)
        # [1차]: 볼록 사각형 강제 여부를 불리언으로 변환합니다.
        # [2차]: 볼록 검사 스위치도 참/거짓으로 명확히 저장합니다.
        self.require_convex = bool(self.require_convex)

    # [1차]: 현재 설정 객체의 모든 필드를 JSON 직렬화 가능한 일반 dict로 변환합니다.
    # [2차]: 설정 카드의 내용을 보고서나 파일에 적기 좋게 표준 서류 양식(딕셔너리)으로 복사합니다.
    def to_dict(self) -> Dict[str, object]:
        """Serialize configuration to a plain dictionary."""
        # [1차]: dataclasses.asdict()로 필드명-값 쌍의 새 딕셔너리를 반환합니다.
        # [2차]: 카드에 적힌 모든 항목을 그대로 옮겨 적은 사본을 돌려줍니다.
        return asdict(self)


# [1차]: 파라미터 딜레마 실험 및 현장 빠른 전환을 위한 5가지 대표 파라미터 프리셋 사전입니다.
# [2차]: '기본 모드', '강한 노이즈 제거', '엣지 연결', 'CLAHE 그림자 보정', '강블러+저임계값' 5가지 레시피 세트입니다.
PRESETS: Dict[str, ScanConfig] = {
    # [1차]: 교과서적 기본 파이프라인 설정(블러 5, Canny 50/150, 모폴로지 미사용).
    # [2차]: 가장 무난한 표준 레시피입니다.
    "default": ScanConfig(blur_kernel=5, canny_low=50, canny_high=150, morph_kernel=0),
    # [1차]: 복잡한 배경/격자 제거용 강노이즈 억제 설정(큰 블러 9, 높은 Canny 80/200, 닫힘 3).
    # [2차]: 시끄러운 배경 무늬를 큰 지우개로 뭉개고 굵은 외곽선만 남기는 레시피입니다.
    "strong_denoise": ScanConfig(
        blur_kernel=9, canny_low=80, canny_high=200, morph_kernel=3, use_clahe=False
    ),
    # [1차]: 끊어진 엣지를 연결하는 모폴로지 닫힘 특화 설정(블러 5, Canny 40/120, 닫힘 5, ε=0.03).
    # [2차]: 중간에 끊긴 점선 테두리를 테이프로 이어 붙여 복구하는 레시피입니다.
    "morph_bridge": ScanConfig(
        blur_kernel=5, canny_low=40, canny_high=120, morph_kernel=5, approx_epsilon_ratio=0.03
    ),
    # [1차]: 손/물체 그림자 보정 특화 설정(CLAHE 활성화, 블러 7, Canny 70/180, 강한 닫힘 9, ε=0.04).
    # [2차]: 그림자 속 숨은 테두리를 국소 조명 보정(CLAHE)과 넓은 테이프로 되살리는 레시피입니다.
    "clahe_shadow_morph": ScanConfig(
        blur_kernel=7,
        canny_low=70,
        canny_high=180,
        morph_kernel=9,
        approx_epsilon_ratio=0.04,
        use_clahe=True,
    ),
    # [1차]: 원거리 흐림/기울어진 문서 특화 설정(강한 블러 11, 낮은 Canny 20/60, 강한 닫힘 9, ε=0.04).
    # [2차]: 큰 지우개로 잔글씨를 지운 뒤 낮은 임계값으로 희미한 원거리 테두리를 잡는 레시피입니다.
    "heavy_blur_morph": ScanConfig(
        blur_kernel=11,
        canny_low=20,
        canny_high=60,
        morph_kernel=9,
        approx_epsilon_ratio=0.04,
        use_clahe=False,
    ),
}
