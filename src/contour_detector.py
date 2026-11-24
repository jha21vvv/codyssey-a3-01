"""Document contour extraction and quadrilateral approximation module."""

# [1차]: 타입 어노테이션의 지연 평가를 활성화합니다.
# [2차]: 아직 뒤에 나올 타입 이름표를 실행 시점까지 잠깐 보류해 둡니다.
from __future__ import annotations

# [1차]: 윤곽선 검출 결과 객체를 깔끔하게 정의하기 위해 dataclass를 가져옵니다.
# [2차]: 검출 결과(4꼭짓점, 면적, 실패 사유 등)를 한 상자에 정리할 틀을 꺼냅니다.
from dataclasses import dataclass

# [1차]: 리스트 및 선택적(Optional) 타입 힌트를 가져옵니다.
# [2차]: "여러 개 목록이거나 값이 없을 수도(None) 있음"을 표시하는 안내 라벨입니다.
from typing import List, Optional

# [1차]: 윤곽선 탐색(findContours), 다각형 근사(approxPolyDP), 시각화 도구를 위해 OpenCV를 가져옵니다.
# [2차]: 스케치북의 닫힌 도형을 찾고 초록 테두리를 그려줄 OpenCV 엔진입니다.
import cv2

# [1차]: 좌표 배열 변환 및 반올림 연산을 위해 NumPy를 가져옵니다.
# [2차]: 꼭짓점 좌표 숫자들을 고속으로 다루기 위한 계산기입니다.
import numpy as np

# [1차]: 면적 필터링·근사 오차·볼록성 규칙이 담긴 ScanConfig를 가져옵니다.
# [2차]: "최소 몇 % 크기부터 문서로 인정할지" 적힌 설정표를 불러옵니다.
from .config import ScanConfig


# [1차]: 엣지 맵에서 문서 사각형을 탐색한 최종 결과와 진단 메타데이터를 담는 데이터클래스입니다.
# [2차]: 가위질 담당이 "네모를 찾았는지, 못 찾았으면 왜 못 찾았는지" 적어내는 작업 보고서입니다.
@dataclass
class ContourDetectionResult:
    """Result of searching for a 4-vertex document boundary in an edge map."""

    # [1차]: 검출된 4개 꼭짓점 (4, 2) float32 배열(실패 시 None).
    # [2차]: 찾아낸 문서의 네 모서리 좌표(못 찾으면 빈값 None)입니다.
    quad_points: Optional[np.ndarray]
    # [1차]: 엣지 맵에서 추출된 전체 윤곽선 총개수입니다.
    # [2차]: 스케치북에 그려진 닫힌 도형이 전부 몇 개였는지 센 숫자입니다.
    contour_count: int
    # [1차]: 면적 필터링(min_area_ratio ~ max_area_ratio)을 통과한 후보 윤곽선 수입니다.
    # [2차]: 너무 작은 먼지나 화면 전체 테두리를 거르고 남은 1차 합격 후보 수입니다.
    candidate_count: int
    # [1차]: 채택된 사각형(또는 최대 후보)의 픽셀 면적입니다.
    # [2차]: 찾은 종이 조각이 차지하는 넓이(픽셀 칸 수)입니다.
    contour_area: float
    # [1차]: 전체 이미지 넓이 대비 검출 윤곽선 면적 비율(0.0 ~ 1.0)입니다.
    # [2차]: 전체 사진에서 이 문서가 몇 % 공간을 차지하는지 나타냅니다.
    area_ratio: float
    # [1차]: 검출된 사각형(초록) 또는 탈락 최대 후보(주황)를 덧그린 시각화 이미지입니다.
    # [2차]: 사람 눈으로 결과를 바로 확인할 수 있게 원본 사진 위에 형광펜으로 테두리를 칠한 사본입니다.
    overlay_image: np.ndarray
    # [1차]: 검출 실패 시 원인 코드 문자열(성공 시 None).
    # [2차]: "윤곽선이 아예 없음", "다 너무 작음", "4각형이 없음" 등 실패 진단명입니다.
    failure_reason: Optional[str] = None


# [1차]: 원본 이미지 사본 위에 검출된 4각형(또는 탈락한 최선 후보 윤곽선)을 그리는 내부 시각화 함수입니다.
# [2차]: "내가 찾은 문서가 바로 여기야!" 하고 원본 사진 사본 위에 초록 선과 빨간 점을 예쁘게 찍어줍니다.
def _draw_quad_overlay(
    image: np.ndarray,
    quad_points: Optional[np.ndarray],
    fallback_contour: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw detected 4-point quadrilateral (or best rejected contour) on a copy of original image."""
    # [1차]: 원본 이미지를 훼손하지 않도록 독립 복사본을 만듭니다.
    # [2차]: 원본 사진에 직접 낙서하지 않고 복사본 위에만 형광펜을 칠합니다.
    overlay = image.copy()
    # [1차]: 입력이 2차원 흑백 이미지이면 컬러 선을 그릴 수 있도록 3채널 BGR로 확장합니다.
    # [2차]: 흑백 도화지에는 초록·빨강 펜이 안 보이므로 컬러 도화지로 바꿔줍니다.
    if overlay.ndim == 2:
        # [1차]: GRAY2BGR 변환을 수행합니다.
        # [2차]: 흑백 밝기를 3채널 컬러 형식으로 복사해 색깔 펜을 쓸 준비를 합니다.
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    # [1차]: 유효한 4개 꼭짓점이 검출된 경우 초록색 테두리와 빨간색 꼭짓점·라벨을 그립니다.
    # [2차]: 문서를 제대로 찾았다면 초록 형광펜으로 네모를 두르고 네 모서리에 빨간 핀을 꽂습니다.
    if quad_points is not None:
        # [1차]: 실수 좌표를 정수 반올림한 뒤 OpenCV polylines 규격 (N, 1, 2)로 변형합니다.
        # [2차]: 소수점 좌표를 화면 픽셀 칸에 맞게 반올림해 OpenCV가 좋아하는 모양으로 접습니다.
        pts_int = np.round(quad_points).astype(np.int32).reshape((-1, 1, 2))
        # [1차]: 닫힌 다각형(isClosed=True)을 두께 3의 초록색(0, 220, 0) 선으로 그립니다.
        # [2차]: 네 모서리를 초록색 굵은 선으로 한 바퀴 연결합니다.
        cv2.polylines(overlay, [pts_int], isClosed=True, color=(0, 220, 0), thickness=3)
        # [1차]: 각 꼭짓점에 붙일 기본 방위 라벨 목록입니다.
        # [2차]: 좌상(TL)·우상(TR)·우하(BR)·좌하(BL) 이름표 4장을 준비합니다.
        labels = ["TL", "TR", "BR", "BL"]
        # [1차]: 4개 꼭짓점 좌표를 순회하며 원형 마커와 텍스트 라벨을 표시합니다.
        # [2차]: 네 귀퉁이마다 차례대로 빨간 점을 찍고 이름표를 달아줍니다.
        for idx, pt in enumerate(pts_int.reshape(4, 2)):
            # [1차]: OpenCV 그리기 함수용 정수 튜플 (x, y)를 만듭니다.
            # [2차]: 점을 찍을 정확한 (가로, 세로) 위치를 정수로 정리합니다.
            center = (int(pt[0]), int(pt[1]))
            # [1차]: 반지름 8픽셀의 채워진 빨간 원(-1)을 그립니다.
            # [2차]: 모서리 한가운데 빨간 압정을 꾹 누릅니다.
            cv2.circle(overlay, center, 8, (0, 0, 255), -1)
            # [1차]: 시인성을 높이기 위해 바깥쪽에 반지름 10의 흰색 테두리 원을 덧그립니다.
            # [2차]: 어두운 배경에서도 잘 보이도록 빨간 압정 둘레에 흰색 링 테두리를 두릅니다.
            cv2.circle(overlay, center, 10, (255, 255, 255), 2)
            # [1차]: 인덱스에 대응하는 라벨 문자열을 선택합니다.
            # [2차]: 현재 모서리에 붙일 이름표 글자를 고릅니다.
            label = labels[idx] if idx < len(labels) else str(idx)
            # [1차]: 꼭짓점 우상단(화면 상단 이탈 방지 max(20, ...))에 노란색 텍스트를 기록합니다.
            # [2차]: 화면 밖으로 글자가 잘리지 않게 살짝 옆에 노란색으로 이름표를 써줍니다.
            cv2.putText(
                overlay,
                label,
                (center[0] + 10, max(20, center[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    # [1차]: 4각형 검출에 실패했지만 참고용 최대 후보 윤곽선이 있으면 주황색으로 표시합니다.
    # [2차]: 네모를 못 찾았을 때는 "가장 컸던 탈락 후보가 이거였어요" 하고 주황색으로 힌트를 남깁니다.
    elif fallback_contour is not None:
        # [1차]: 단일 후보 윤곽선을 두께 2의 주황색(0, 80, 255)으로 그립니다.
        # [2차]: 탈락한 1등 후보 테두리를 주황색 펜으로 표시해 원인 분석을 돕습니다.
        cv2.drawContours(overlay, [fallback_contour], -1, (0, 80, 255), 2)

    # [1차]: 시각화가 완료된 오버레이 이미지를 반환합니다.
    # [2차]: 형광펜 표시가 끝난 확인용 사진을 돌려줍니다.
    return overlay


# [1차]: Canny 엣지 맵에서 닫힌 윤곽선을 추출하고 면적·4꼭짓점·볼록성 규칙으로 문서 사각형을 찾습니다.
# [2차]: 하얀 외곽선 그림 속에서 "가장 크고 반듯한 네모 종이"를 골라내는 핵심 탐색 함수입니다.
def find_document_contour(
    edges: np.ndarray,
    original_image: np.ndarray,
    config: ScanConfig,
) -> ContourDetectionResult:
    """Extract closed contours, filter by area, and select the largest 4-vertex quadrilateral."""
    # [1차]: 입력 엣지 배열이 비어 있으면 즉시 ValueError를 발생시킵니다.
    # [2차]: 밑그림(엣지 사진)이 비어 있으면 탐색을 멈추고 알립니다.
    if edges is None or edges.size == 0:
        # [1차]: 빈 엣지 입력 예외를 던집니다.
        # [2차]: "밑그림이 비어 있어서 문서를 찾을 수 없어요!"라고 보고합니다.
        raise ValueError("엣지 이미지가 비어 있습니다.")

    # [1차]: 이미지 세로(height)·가로(width) 크기와 전체 픽셀 면적을 계산합니다.
    # [2차]: 전체 도화지의 가로×세로 넓이를 먼저 계산해 비율 기준을 세웁니다.
    height, width = edges.shape[:2]
    # [1차]: 전체 면적을 float으로 변환해 이후 비율 나눗셈의 정밀도를 확보합니다.
    # [2차]: 전체 도화지 칸 수를 소수점 계산용 숫자로 준비합니다.
    total_area = float(height * width)
    # [1차]: 설정된 최소 면적 비율을 픽셀 단위 하한 면적으로 환산합니다.
    # [2차]: "최소 이 크기(예: 전체의 5%)는 넘어야 문서지!" 하는 최소 넓이 컷라인입니다.
    min_area = total_area * config.min_area_ratio
    # [1차]: 설정된 최대 면적 비율을 픽셀 단위 상한 면적으로 환산합니다.
    # [2차]: "전체 화면(98% 이상)을 다 덮는 테두리는 사진 프레임이니 제외!" 하는 상한선입니다.
    max_area = total_area * config.max_area_ratio

    # [1차]: RETR_LIST로 모든 계층의 닫힌 윤곽선을 찾고 CHAIN_APPROX_SIMPLE로 직선 중복 점을 압축합니다.
    # [2차]: 책상 테두리 안쪽에 놓인 문서까지 놓치지 않도록 모든 닫힌 선을 전부 수집합니다.
    contours_raw, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # [1차]: 추출된 모든 윤곽선을 면적(cv2.contourArea) 내림차순으로 정렬합니다.
    # [2차]: 문서가 보통 화면에서 가장 큰 물체이므로 덩치가 큰 순서대로 줄을 세웁니다.
    contours: List[np.ndarray] = sorted(contours_raw, key=cv2.contourArea, reverse=True)

    # [1차]: 닫힌 윤곽선이 단 하나도 검출되지 않은 경우 실패 결과를 구조화해 반환합니다.
    # [2차]: 민무늬 도화지처럼 선이 아예 없으면 에러로 멈추지 않고 "선 없음" 보고서를 냅니다.
    if not contours:
        # [1차]: quad_points=None과 failure_reason="no_contours_found"를 담아 반환합니다.
        # [2차]: "닫힌 선이 0개입니다"라는 진단서를 돌려줍니다.
        return ContourDetectionResult(
            quad_points=None,
            contour_count=0,
            candidate_count=0,
            contour_area=0.0,
            area_ratio=0.0,
            overlay_image=_draw_quad_overlay(original_image, None),
            failure_reason="no_contours_found",
        )

    # [1차]: 면적 하한~상한 범위 안에 들어오는 유효 후보 윤곽선만 선별합니다.
    # [2차]: 너무 작은 먼지와 너무 큰 화면 테두리를 걸러내고 1차 합격자만 모읍니다.
    candidates: List[np.ndarray] = []
    # [1차]: 면적 큰 순으로 정렬된 윤곽선을 순회하며 범위 조건을 검사합니다.
    # [2차]: 줄 서 있는 도형들을 하나씩 키 재기 통과선에 대어봅니다.
    for contour in contours:
        # [1차]: 현재 윤곽선 내부의 픽셀 면적을 계산합니다.
        # [2차]: 이 도형의 넓이가 몇 칸인지 측정합니다.
        area = cv2.contourArea(contour)
        # [1차]: min_area <= area <= max_area를 만족하면 후보 목록에 추가합니다.
        # [2차]: 적당한 문서 크기 범위 안에 들면 1차 합격자 명단에 올립니다.
        if min_area <= area <= max_area:
            # [1차]: 합격 윤곽선을 candidates 리스트에 추가합니다.
            # [2차]: 통과한 도형을 후보 바구니에 담습니다.
            candidates.append(contour)

    # [1차]: 면적 필터를 통과한 후보가 하나도 없으면 가장 컸던 윤곽선 정보와 함께 실패를 반환합니다.
    # [2차]: 외곽선이 끊어져서 다 조그만 조각으로 쪼개졌을 때 "전부 면적 미달"로 정확히 진단합니다.
    if not candidates:
        # [1차]: 진단용으로 가장 큰 윤곽선의 면적을 기록합니다.
        # [2차]: 탈락한 조각 중 제일 컸던 녀석의 크기를 참고용으로 적어둡니다.
        largest_area = float(cv2.contourArea(contours[0]))
        # [1차]: 실패 사유 "area_filter_rejected_all"과 주황색 폴백 오버레이를 반환합니다.
        # [2차]: "모든 윤곽선이 최소 면적 기준을 넘지 못했어요"라고 보고합니다.
        return ContourDetectionResult(
            quad_points=None,
            contour_count=len(contours),
            candidate_count=0,
            contour_area=largest_area,
            area_ratio=largest_area / total_area if total_area > 0 else 0.0,
            overlay_image=_draw_quad_overlay(original_image, None, fallback_contour=contours[0]),
            failure_reason="area_filter_rejected_all",
        )

    # [1차]: 면적 상위 최대 15개 후보에 대해 Douglas-Peucker 다각형 근사(approxPolyDP)를 수행합니다.
    # [2차]: 덩치 큰 순서대로 상위 15개 후보를 꺼내 "꼭짓점이 딱 4개인 반듯한 네모인가?" 검사합니다.
    for contour in candidates[:15]:
        # [1차]: 닫힌 윤곽선(True)의 전체 둘레 길이를 계산합니다.
        # [2차]: 도형 테두리를 줄자로 한 바퀴 잰 총 길이를 구합니다.
        perimeter = cv2.arcLength(contour, True)
        # [1차]: 둘레가 0 이하인 퇴화 윤곽선은 건너뜁니다.
        # [2차]: 길이가 0인 유령 도형은 무시하고 다음으로 넘어갑니다.
        if perimeter <= 0:
            continue
        # [1차]: 둘레의 approx_epsilon_ratio 비율을 최대 허용 오차로 두고 다각형 꼭짓점을 단순화합니다.
        # [2차]: 살짝 울퉁불퉁한 테두리를 직선으로 팽팽하게 펴서 꼭짓점이 몇 개 남는지 세어봅니다.
        approx = cv2.approxPolyDP(contour, config.approx_epsilon_ratio * perimeter, True)
        # [1차]: 근사화된 꼭짓점이 정확히 4개인지 확인합니다.
        # [2차]: "꼭짓점이 딱 4개? 오, 사각형 문서 후보 발견!" 하고 확인합니다.
        if len(approx) == 4:
            # [1차]: 볼록성 검사가 켜져 있을 때 오목(별/부메랑 모양) 사각형은 탈락시킵니다.
            # [2차]: 안으로 움푹 파인 이상한 4각형은 정상 종이가 아니므로 거절합니다.
            if config.require_convex and not cv2.isContourConvex(approx):
                continue
            # [1차]: 근사화된 4각형 자체의 면적도 여전히 최소 면적 이상인지 재검증합니다.
            # [2차]: 직선으로 펴고 났더니 홀쭉해져서 너무 작아지지 않았는지 한 번 더 확인합니다.
            area = float(cv2.contourArea(approx))
            # [1차]: 면적 기준 미달 시 다음 후보로 넘어갑니다.
            # [2차]: 너무 작아졌으면 아쉽지만 탈락시키고 다음 후보를 봅니다.
            if area < min_area:
                continue
            # [1차]: (4, 1, 2) 형태를 (4, 2) float32 좌표 행렬로 평탄화합니다.
            # [2차]: 불필요한 중첩 괄호를 벗겨 깔끔한 4행 2열(x, y) 좌표표로 정리합니다.
            quad = approx.reshape(4, 2).astype(np.float32)
            # [1차]: 최종 검출된 사각형과 시각화 오버레이를 성공 결과로 반환합니다.
            # [2차]: "최고의 문서 사각형을 찾았습니다!" 하고 좌표와 초록 표시 사진을 넘겨줍니다.
            return ContourDetectionResult(
                quad_points=quad,
                contour_count=len(contours),
                candidate_count=len(candidates),
                contour_area=area,
                area_ratio=area / total_area,
                overlay_image=_draw_quad_overlay(original_image, quad),
                failure_reason=None,
            )

    # [1차]: 면적 후보들은 있었으나 모두 4꼭짓점 볼록 사각형 조건을 통과하지 못한 경우 실패를 반환합니다.
    # [2차]: 큰 도형은 있었지만 손가락 가림·구겨짐 등으로 5각형·6각형이 되어 사각형 찾기에 실패했음을 보고합니다.
    best_area = float(cv2.contourArea(candidates[0]))
    # [1차]: 실패 사유 "no_4_vertex_quad_found"와 1순위 탈락 후보 오버레이를 반환합니다.
    # [2차]: "4꼭짓점 사각형을 찾지 못했습니다"라는 최종 진단서를 돌려줍니다.
    return ContourDetectionResult(
        quad_points=None,
        contour_count=len(contours),
        candidate_count=len(candidates),
        contour_area=best_area,
        area_ratio=best_area / total_area,
        overlay_image=_draw_quad_overlay(original_image, None, fallback_contour=candidates[0]),
        failure_reason="no_4_vertex_quad_found",
    )
