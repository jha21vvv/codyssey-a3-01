"""Manual geometric 4-point coordinate ordering and perspective transformation module.

Zero external geometry libraries (no scipy.spatial, no shapely) are used.
All coordinate sorting is implemented strictly with Python built-ins and NumPy.
"""

# [1차]: 타입 힌트 문자열 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 나중에 해석하도록 미리 켜둡니다.
from __future__ import annotations

# [1차]: 무게중심 기준 편각(atan2) 계산을 위해 파이썬 표준 math 모듈을 가져옵니다.
# [2차]: 외부 기하학 라이브러리 없이 내장 각도기(math.atan2)만으로 시계 방향 각도를 잽니다.
import math

# [1차]: 반환 튜플 타입 명시를 위해 Tuple을 가져옵니다.
# [2차]: "여러 값을 묶음(튜플)으로 돌려준다"고 표시하는 라벨입니다.
from typing import Tuple

# [1차]: 투시 변환 행렬 계산(getPerspectiveTransform), 워핑(warpPerspective), IoU 마스크(fillPoly)를 위해 cv2를 가져옵니다.
# [2차]: 비스듬한 사다리꼴을 정면 직사각형으로 펴주는 OpenCV 기하 변환 도구입니다.
import cv2

# [1차]: 벡터 노름, 내적, 롤(roll), 논리 연산을 위해 NumPy를 가져옵니다.
# [2차]: 좌표 덧셈·거리 계산·행렬 조작을 순식간에 처리하는 숫자 엔진입니다.
import numpy as np


# [1차]: 가우스 신발끈 공식(Shoelace Formula)을 순수 NumPy 벡터 내적으로 구현해 다각형 넓이를 구합니다.
# [2차]: 네 점이 한 줄에 서서 넓이가 0인 '찌그러진 직선'은 아닌지 신발끈을 X자로 묶듯 넓이를 계산해 검사합니다.
def _polygon_area_shoelace(pts: np.ndarray) -> float:
    """Compute polygon area using the Gauss shoelace formula via pure NumPy."""
    # [1차]: 모든 꼭짓점의 x 좌표 벡터를 슬라이싱합니다.
    # [2차]: 네 모서리의 가로 위치만 한 줄로 모읍니다.
    x = pts[:, 0]
    # [1차]: 모든 꼭짓점의 y 좌표 벡터를 슬라이싱합니다.
    # [2차]: 네 모서리의 세로 위치만 한 줄로 모읍니다.
    y = pts[:, 1]
    # [1차]: 0.5 * |Σ(x_i * y_{i+1} - y_i * x_{i+1})| 공식을 np.dot과 np.roll(-1)로 계산합니다.
    # [2차]: 이웃한 점끼리 엇갈려 곱하고 뺀 뒤 절반을 취하면 다각형의 정확한 넓이가 나옵니다.
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# [1차]: 임의 순서의 4개 꼭짓점을 외부 기하 라이브러리 없이 [좌상, 우상, 우하, 좌하] 시계 방향으로 정렬합니다.
# [2차]: 뒤죽박죽 섞인 네 모서리에 "0:좌상 → 1:우상 → 2:우하 → 3:좌하" 자리 번호표를 절대 꼬이지 않게 붙여줍니다.
def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner coordinates strictly as [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Uses centroid polar-angle sorting in screen coordinates (y downward) with
    deterministic lexicographical tie-breaking so even 45-degree rotated diamond
    documents never suffer from duplicate vertex selection or self-intersection.
    """
    # [1차]: 입력을 float32 NumPy 배열로 변환하고 (-1, 2) 형태로 2차원 좌표 목록으로 정규화합니다.
    # [2차]: (4, 1, 2)든 리스트든 어떤 형태로 들어와도 깔끔한 "점 4개 × (x, y)" 표로 펼칩니다.
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    # [1차]: 정확히 4개의 2차원 점인지 형상을 검증합니다.
    # [2차]: 꼭짓점이 3개나 5개면 사각형이 아니므로 바로 거절합니다.
    if arr.shape != (4, 2):
        # [1차]: 잘못된 입력 형상을 담아 ValueError를 발생시킵니다.
        # [2차]: "네 모서리가 정확히 4개여야 합니다!"라고 알려줍니다.
        raise ValueError(f"정확히 4개의 2차원 좌표가 필요합니다. 입력 형태: {arr.shape}")

    # [1차]: 모든 좌표가 유한 실수(NaN/Inf 아님)인지 검사합니다.
    # [2차]: 숫자가 깨진 값(NaN/무한대)이 섞여 있으면 계산이 터지기 전에 막아줍니다.
    if not np.all(np.isfinite(arr)):
        # [1차]: 비정상 부동소수점 좌표에 대해 ValueError를 발생시킵니다.
        # [2차]: "이상한 숫자(NaN/무한대)가 들어 있어 정렬할 수 없어요!"라고 경고합니다.
        raise ValueError("좌표에 NaN 또는 무한대 값이 포함되어 있습니다.")

    # [1차]: 4개 꼭짓점 x 좌표의 산술평균으로 무게중심 cx를 구합니다.
    # [2차]: 네 모서리 한가운데 서 있을 중심점의 가로 위치를 찾습니다.
    centroid_x = float(np.mean(arr[:, 0]))
    # [1차]: 4개 꼭짓점 y 좌표의 산술평균으로 무게중심 cy를 구합니다.
    # [2차]: 네 모서리 한가운데 서 있을 중심점의 세로 위치를 찾습니다.
    centroid_y = float(np.mean(arr[:, 1]))

    # [1차]: 각 점의 (편각, x+y, y, x, 원본인덱스) 튜플을 저장할 리스트를 초기화합니다.
    # [2차]: 각 모서리가 중심에서 몇 시 방향에 있는지 각도표를 적을 메모장을 준비합니다.
    indexed = []
    # [1차]: 4개 점을 순회하며 화면 좌표계(y 아래 방향) 기준 시계 방향 편각 atan2를 계산합니다.
    # [2차]: 중심에 서서 각 모서리를 손가락으로 가리키며 각도기 눈금을 하나씩 적습니다.
    for idx in range(4):
        # [1차]: 현재 점의 x 좌표를 파이썬 float으로 꺼냅니다.
        # [2차]: 현재 모서리의 가로 좌표를 읽습니다.
        x = float(arr[idx, 0])
        # [1차]: 현재 점의 y 좌표를 파이썬 float으로 꺼냅니다.
        # [2차]: 현재 모서리의 세로 좌표를 읽습니다.
        y = float(arr[idx, 1])
        # [1차]: atan2(y - cy, x - cx)는 이미지 좌표계(y↓)에서 -π(좌)→-π/2(상)→0(우)→+π/2(하) 시계 방향입니다.
        # [2차]: 화면에서는 아래로 갈수록 y가 커지므로, 각도가 작은 순서대로 정렬하면 자연스럽게 시계 방향이 됩니다!
        angle = math.atan2(y - centroid_y, x - centroid_x)
        # [1차]: 동률(Tie) 발생 시 결정론적 순서를 보장하도록 (angle, x+y, y, x, idx)를 저장합니다.
        # [2차]: 각도가 똑같은 희귀 상황에서도 순서가 흔들리지 않게 예비 기준(x+y, y, x)까지 적어둡니다.
        indexed.append((angle, x + y, y, x, idx))

    # [1차]: 편각 오름차순(시계 방향) 및 사전순 타이 브레이킹으로 4개 점을 정렬합니다.
    # [2차]: 각도기 눈금이 작은 점부터 차례대로 세워 시계 방향 1바퀴 순서를 확정합니다.
    indexed.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    # [1차]: 정렬된 인덱스에 따라 4개 고유 좌표를 새 NumPy 배열로 재구성합니다.
    # [2차]: 확정된 시계 방향 순서대로 네 모서리 좌표를 새 표에 옮겨 적습니다.
    cw_points = np.array([arr[item[4]] for item in indexed], dtype=np.float32)

    # [1차]: 신발끈 넓이가 0.001 미만이면 4점이 일직선이거나 중복점이므로 거부합니다.
    # [2차]: 네 점을 이어봤더니 종이가 아니라 얇은 실(넓이 0)이면 워핑할 수 없으므로 차단합니다.
    if _polygon_area_shoelace(cw_points) < 1e-3:
        # [1차]: 퇴화 사각형 오류를 발생시킵니다.
        # [2차]: "네 점이 한 줄로 겹쳐 있어 사각형이 안 됩니다!"라고 알립니다.
        raise ValueError("4개의 점이 일직선상에 있거나 중복되어 유효한 사각형을 형성하지 못합니다.")

    # [1차]: 시계 방향 4점 중 원점(0,0)에 가장 가까운 좌상단(TL) 점의 인덱스를 사전순 타이브레이킹으로 선택합니다.
    # [2차]: 시계 방향 고리에서 "어디가 1번(좌상단) 출발점이지?"를 x+y가 가장 작은 점으로 딱 집어냅니다.
    tl_idx = min(
        range(4),
        key=lambda i: (float(cw_points[i, 0] + cw_points[i, 1]), float(cw_points[i, 1]), float(cw_points[i, 0])),
    )
    # [1차]: np.roll로 시계 방향 순서를 유지한 채 tl_idx가 0번이 되도록 순환 이동합니다.
    # [2차]: 원탁에 둘러앉은 4명의 순서는 그대로 두고, 좌상단(TL)이 맨 앞자리에 오도록 회전판만 돌려줍니다.
    ordered = np.roll(cw_points, -tl_idx, axis=0)
    # [1차]: OpenCV getPerspectiveTransform 규격에 맞는 float32 (4, 2) 배열을 반환합니다.
    # [2차]: [좌상, 우상, 우하, 좌하] 순서가 100% 보장된 최종 좌표표를 건네줍니다.
    return ordered.astype(np.float32)


# [1차]: 보고서 및 단위 테스트에서 45° 마름모 동률 붕괴를 대조 시연하기 위한 기존 sum/diff 방식입니다.
# [2차]: "왜 단순 x+y / y-x 방식이 45도 기울어진 종이에서 망가지는지" 비교 증명용으로 남겨둔 구형 방식입니다.
def order_points_naive_sum_diff(pts: np.ndarray) -> np.ndarray:
    """Classic sum/diff heuristic kept purely for edge-case contrast in tests and report."""
    # [1차]: 입력을 (4, 2) float32 배열로 변환합니다.
    # [2차]: 4개 모서리 좌표를 표로 정리합니다.
    arr = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    # [1차]: 정렬 결과를 담을 4×2 0행렬을 생성합니다.
    # [2차]: 빈 자리 4칸을 준비합니다.
    rect = np.zeros((4, 2), dtype=np.float32)
    # [1차]: 각 점의 x+y 합을 계산합니다.
    # [2차]: 가로+세로 합계를 구합니다(45도 마름모에서는 여기서 동률이 터집니다!).
    s = arr.sum(axis=1)
    # [1차]: 각 점의 y-x 차이를 계산합니다.
    # [2차]: 세로-가로 차이를 구합니다.
    d = np.diff(arr, axis=1).reshape(-1)
    # [1차]: x+y 최솟값을 TL(0번)에 배정합니다.
    # [2차]: 합계가 제일 작은 점을 좌상단에 넣습니다.
    rect[0] = arr[int(np.argmin(s))]
    # [1차]: x+y 최댓값을 BR(2번)에 배정합니다.
    # [2차]: 합계가 제일 큰 점을 우하단에 넣습니다.
    rect[2] = arr[int(np.argmax(s))]
    # [1차]: y-x 최솟값을 TR(1번)에 배정합니다(동률 시 TL과 중복 선택 발생 가능).
    # [2차]: 차이가 제일 작은 점을 우상단에 넣습니다(45도에서는 좌상단과 같은 점을 또 고릅니다!).
    rect[1] = arr[int(np.argmin(d))]
    # [1차]: y-x 최댓값을 BL(3번)에 배정합니다.
    # [2차]: 차이가 제일 큰 점을 좌하단에 넣습니다.
    rect[3] = arr[int(np.argmax(d))]
    # [1차]: 단순 정렬 결과를 반환합니다.
    # [2차]: 비교 실험용 구형 정렬 결과를 돌려줍니다.
    return rect


# [1차]: 정렬된 4꼭짓점의 윗변·아랫변 및 좌변·우변 유클리드 거리로부터 출력 직사각형 (가로, 세로)를 구합니다.
# [2차]: 비스듬한 종이를 폈을 때 글씨가 홀쭉하거나 뚱뚱해지지 않도록 가로·세로 중 더 긴 변 길이로 크기를 정합니다.
def compute_output_dimensions(ordered_pts: np.ndarray) -> Tuple[int, int]:
    """Calculate target rectangle (width, height) from Euclidean edge lengths."""
    # [1차]: 정렬된 4개 좌표를 좌상(tl), 우상(tr), 우하(br), 좌하(bl)로 언패킹합니다.
    # [2차]: 네 모서리에 각각 이름표를 붙여 꺼냅니다.
    tl, tr, br, bl = ordered_pts
    # [1차]: 윗변 길이 ||tr - tl||을 L2 유클리드 노름으로 계산합니다.
    # [2차]: 피타고라스 정리로 종이 윗변의 실제 길이를 잽니다.
    width_top = float(np.linalg.norm(tr - tl))
    # [1차]: 아랫변 길이 ||br - bl||을 L2 유클리드 노름으로 계산합니다.
    # [2차]: 피타고라스 정리로 종이 아랫변의 실제 길이를 잽니다.
    width_bottom = float(np.linalg.norm(br - bl))
    # [1차]: 두 가로 길이 중 최댓값을 반올림하되 최소 1픽셀 이상을 보장합니다.
    # [2차]: 윗변과 아랫변 중 더 긴 길이를 펴진 종이의 가로 크기로 선택합니다.
    max_width = max(int(round(max(width_top, width_bottom))), 1)

    # [1차]: 오른변 길이 ||tr - br||을 계산합니다.
    # [2차]: 종이 오른쪽 변의 길이를 잽니다.
    height_right = float(np.linalg.norm(tr - br))
    # [1차]: 왼변 길이 ||tl - bl||을 계산합니다.
    # [2차]: 종이 왼쪽 변의 길이를 잽니다.
    height_left = float(np.linalg.norm(tl - bl))
    # [1차]: 두 세로 길이 중 최댓값을 반올림하되 최소 1픽셀 이상을 보장합니다.
    # [2차]: 왼변과 오른변 중 더 긴 길이를 펴진 종이의 세로 크기로 선택합니다.
    max_height = max(int(round(max(height_right, height_left))), 1)
    # [1차]: 최종 출력 이미지 크기 (max_width, max_height) 튜플을 반환합니다.
    # [2차]: 새로 만들 직사각형 도화지의 (가로, 세로) 치수를 건네줍니다.
    return max_width, max_height


# [1차]: 4점 좌표 정렬 → 3×3 투시 변환 행렬(Homography) 산출 → 정면 직사각형 워핑을 수행합니다.
# [2차]: 비스듬하게 찍힌 사다리꼴 문서를 네 모서리에서 팽팽하게 잡아당겨 정면 스캔 사진으로 펴줍니다.
def four_point_transform(
    image: np.ndarray,
    pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp a 4-point quadrilateral region into a flat upright top-down rectangle."""
    # [1차]: 입력 이미지가 비어 있으면 ValueError를 발생시킵니다.
    # [2차]: 다림질할 원본 사진이 비어 있으면 즉시 알려줍니다.
    if image is None or image.size == 0:
        # [1차]: 빈 이미지 예외를 던집니다.
        # [2차]: "변환할 원본 사진이 없어요!"라고 보고합니다.
        raise ValueError("변환할 원본 이미지가 비어 있습니다.")

    # [1차]: 수동 기하 정렬 함수로 4개 좌표를 [TL, TR, BR, BL] 순서로 확정합니다.
    # [2차]: 먼저 네 모서리에 1~4번 자리 번호표를 정확히 붙입니다.
    ordered = order_points(pts)
    # [1차]: 출력 직사각형의 (가로, 세로) 픽셀 크기를 계산합니다.
    # [2차]: 종이를 폈을 때의 최종 가로·세로 크기를 잽니다.
    max_width, max_height = compute_output_dimensions(ordered)

    # [1차]: 변환 후 네 꼭짓점이 도착할 반듯한 직사각형 목표 좌표 4개를 정의합니다.
    # [2차]: "좌상은 (0,0), 우상은 (가로-1, 0)..."으로 이동할 목적지 도화지 네 귀퉁이를 찍습니다.
    dst = np.array(
        [
            [0.0, 0.0],
            [float(max_width - 1), 0.0],
            [float(max_width - 1), float(max_height - 1)],
            [0.0, float(max_height - 1)],
        ],
        dtype=np.float32,
    )

    # [1차]: 4쌍의 대응점으로부터 8자유도 3×3 투시 변환 행렬(Homography) M을 구합니다.
    # [2차]: 비스듬한 점들을 반듯한 직사각형으로 순간이동시키는 마법의 3×3 좌표 공식(행렬)을 계산합니다.
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    # [1차]: 행렬 M을 모든 출력 픽셀에 역매핑 적용해 반듯하게 펴진 이미지를 생성합니다.
    # [2차]: 계산한 공식을 사진 전체 픽셀에 적용해 기울어진 문서를 정면에서 찍은 것처럼 반듯하게 폅니다.
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    # [1차]: (펴진 이미지, 정렬된 4꼭짓점, 3×3 변환 행렬) 튜플을 반환합니다.
    # [2차]: 반듯해진 문서 사진과 정렬된 모서리, 그리고 사용한 변환 공식까지 세트로 돌려줍니다.
    return warped, ordered, matrix


# [1차]: 예측 사각형과 정답 사각형의 IoU(교집합/합집합 비율)를 순수 OpenCV 마스크 래스터화로 계산합니다.
# [2차]: 내 스캐너가 찾은 종이 영역과 실제 정답 종이 영역이 몇 %나 정확히 겹치는지(0.0~1.0) 채점합니다.
def compute_quad_iou(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    image_shape: Tuple[int, int],
) -> float:
    """Compute Intersection-over-Union between two 4-point quads via binary rasterization."""
    # [1차]: 래스터 마스크 크기를 원본 이미지의 (높이, 너비) 정수로 꺼냅니다.
    # [2차]: 채점용 투명 모눈종이 크기를 원본 사진과 똑같이 맞춥니다.
    height, width = int(image_shape[0]), int(image_shape[1])
    # [1차]: 예측 영역을 칠할 0 초기화 이진 마스크 배열을 생성합니다.
    # [2차]: 내 스캐너가 찾은 네모를 색칠할 빈 모눈종이 1장을 준비합니다.
    pred_mask = np.zeros((height, width), dtype=np.uint8)
    # [1차]: 정답 영역을 칠할 0 초기화 이진 마스크 배열을 생성합니다.
    # [2차]: 실제 정답 네모를 색칠할 빈 모눈종이 1장을 준비합니다.
    gt_mask = np.zeros((height, width), dtype=np.uint8)

    # [1차]: 예측 4점을 시계 방향 정렬한 뒤 정수 다각형 형태로 변환합니다.
    # [2차]: 내 네모를 꼬임 없이 깔끔하게 색칠할 수 있도록 순서를 맞춥니다.
    pred_poly = np.round(order_points(pred_pts)).astype(np.int32).reshape((-1, 1, 2))
    # [1차]: 정답 4점도 시계 방향 정렬한 뒤 정수 다각형 형태로 변환합니다.
    # [2차]: 정답 네모도 깔끔하게 색칠할 수 있도록 순서를 맞춥니다.
    gt_poly = np.round(order_points(gt_pts)).astype(np.int32).reshape((-1, 1, 2))

    # [1차]: 예측 다각형 내부 픽셀을 1로 채웁니다.
    # [2차]: 첫 번째 모눈종이에 내 네모 안쪽을 색칠합니다.
    cv2.fillPoly(pred_mask, [pred_poly], 1)
    # [1차]: 정답 다각형 내부 픽셀을 1로 채웁니다.
    # [2차]: 두 번째 모눈종이에 정답 네모 안쪽을 색칠합니다.
    cv2.fillPoly(gt_mask, [gt_poly], 1)

    # [1차]: 두 마스크가 모두 1인 픽셀 수(교집합 면적)를 합산합니다.
    # [2차]: 두 모눈종이를 겹쳐서 색이 동시에 칠해진 칸 수(교집합)를 셉니다.
    intersection = int(np.logical_and(pred_mask, gt_mask).sum())
    # [1차]: 두 마스크 중 하나라도 1인 픽셀 수(합집합 면적)를 합산합니다.
    # [2차]: 둘 중 하나라도 색이 칠해진 전체 칸 수(합집합)를 셉니다.
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    # [1차]: 합집합이 0인 극단적 경우 0 나눗셈을 방어하여 0.0을 반환합니다.
    # [2차]: 색칠된 칸이 아예 없으면 에러 내지 않고 0점으로 처리합니다.
    if union == 0:
        return 0.0
    # [1차]: 교집합 / 합집합 비율을 float으로 반환합니다(완벽 일치 시 1.0).
    # [2차]: 겹친 비율(1.0이면 100% 완벽 일치)을 최종 점수로 돌려줍니다.
    return float(intersection) / float(union)


# [1차]: 정렬된 예측 4꼭짓점과 정답 4꼭짓점 사이의 평균 유클리드 픽셀 거리 오차를 계산합니다.
# [2차]: 내 스캐너가 찍은 4개 모서리 핀이 정답 모서리에서 평균 몇 픽셀 빗나갔는지 자로 잽니다.
def compute_corner_error(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """Compute mean Euclidean pixel distance between ordered predicted and ground-truth corners."""
    # [1차]: 예측 좌표를 [TL, TR, BR, BL] 순서로 정렬합니다.
    # [2차]: 내 네 모서리를 좌상부터 차례로 줄 세웁니다.
    pred_ordered = order_points(pred_pts)
    # [1차]: 정답 좌표를 동일한 [TL, TR, BR, BL] 순서로 정렬합니다.
    # [2차]: 정답 네 모서리도 같은 순서로 줄 세워 짝을 맞춥니다.
    gt_ordered = order_points(gt_pts)
    # [1차]: 각 대응 꼭짓점 쌍의 L2 거리(픽셀)를 행 방향(axis=1)으로 구합니다.
    # [2차]: 좌상끼리, 우상끼리, 우하끼리, 좌하끼리 떨어진 거리를 각각 계산합니다.
    dists = np.linalg.norm(pred_ordered - gt_ordered, axis=1)
    # [1차]: 4개 꼭짓점 거리 오차의 산술 평균을 float으로 반환합니다.
    # [2차]: 네 모서리의 평균 오차(픽셀)를 성적표로 돌려줍니다.
    return float(np.mean(dists))
