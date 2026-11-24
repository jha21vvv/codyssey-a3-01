"""End-to-end document scanning pipeline orchestrator and stage montage builder."""

# [1차]: 타입 어노테이션 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 나중에 해석하도록 미리 켜둡니다.
from __future__ import annotations

# [1차]: 파이프라인 전체 실행 결과 구조체 정의를 위해 dataclass를 가져옵니다.
# [2차]: 스캔의 모든 중간·최종 결과물을 한 번에 담을 결과 상자 틀을 준비합니다.
from dataclasses import dataclass

# [1차]: 출력 폴더·파일 경로 처리를 위해 Path를 가져옵니다.
# [2차]: 저장할 폴더와 파일 이름을 깔끔하게 다루는 주소 객체입니다.
from pathlib import Path

# [1차]: 딕셔너리·선택적 값·튜플 타입 힌트를 가져옵니다.
# [2차]: 함수 입력과 출력 모양을 미리 알려주는 안내 라벨들입니다.
from typing import Dict, Optional, Tuple

# [1차]: 윤곽선 재시각화 및 6패널 몽타주 리사이즈·텍스트 렌더링을 위해 OpenCV를 가져옵니다.
# [2차]: 결과물 6장을 한 장의 액자(몽타주)로 예쁘게 편집하는 도구입니다.
import cv2

# [1차]: 배열 합치기(hstack/vstack) 및 캔버스 생성을 위해 NumPy를 가져옵니다.
# [2차]: 사진 6장을 바둑판처럼 가로·세로로 이어 붙이는 행렬 도구입니다.
import numpy as np

# [1차]: 파이프라인 설정 객체 ScanConfig를 가져옵니다.
# [2차]: 다이얼 눈금 설정표를 불러옵니다.
from .config import ScanConfig

# [1차]: 윤곽선 추출 결과 구조체와 문서 사각형 탐색 함수를 가져옵니다.
# [2차]: 가위질 담당 직원을 불러옵니다.
from .contour_detector import ContourDetectionResult, find_document_contour

# [1차]: 수동 4점 정렬 + 투시 변환 함수를 가져옵니다.
# [2차]: 종이 다림질 담당 장인을 불러옵니다.
from .geometry import four_point_transform

# [1차]: 경로 타입 및 한글 안전 이미지 로드·저장 함수를 가져옵니다.
# [2차]: 다국어 택배 기사를 불러옵니다.
from .io_utils import PathLike, load_image, save_image

# [1차]: 적응형 이진화 후처리 함수를 가져옵니다.
# [2차]: 흑백 스캔 복사기 담당을 불러옵니다.
from .postprocessor import postprocess_scan

# [1차]: 전처리 중간 결과 구조체와 전처리 오케스트레이션 함수를 가져옵니다.
# [2차]: 스케치 밑그림 담당을 불러옵니다.
from .preprocessor import PreprocessArtifacts, preprocess_image


# [1차]: 단일 문서 스캔의 모든 단계별 이미지, 4점 좌표, 변환 행렬, 진단 정보를 보관하는 데이터클래스입니다.
# [2차]: "원본부터 최종 스캔본까지" 6단계 공정 사진과 성적표를 모두 담아두는 종합 폴더입니다.
@dataclass
class ScanResult:
    """Complete artifacts produced by a single document scanning run."""

    # [1차]: 문서 4꼭짓점 검출 및 변환 성공 여부 플래그입니다.
    # [2차]: 스캔이 성공했는지(True) 실패했는지(False) 적는 도장입니다.
    detected: bool
    # [1차]: 원본 컬러 이미지 사본입니다.
    # [2차]: 처음 들어온 날것 그대로의 사진입니다.
    original: np.ndarray
    # [1차]: 1채널 흑백 변환 이미지입니다.
    # [2차]: 색깔을 뺀 흑백 사진입니다.
    grayscale: np.ndarray
    # [1차]: 가우시안 블러가 적용된 이미지입니다.
    # [2차]: 자잘한 노이즈를 지우개로 흐린 사진입니다.
    blurred: np.ndarray
    # [1차]: Canny 엣지 검출 결과 이진 이미지입니다.
    # [2차]: 테두리만 하얀 분필로 따낸 밑그림입니다.
    edges: np.ndarray
    # [1차]: 정렬된 4꼭짓점(TL/TR/BR/BL)과 초록 테두리가 표시된 시각화 이미지입니다.
    # [2차]: 찾은 모서리에 순서 이름표까지 붙인 확인용 사진입니다.
    contour_overlay: np.ndarray
    # [1차]: 시계 방향으로 정렬된 4꼭짓점 (4, 2) 좌표(실패 시 None).
    # [2차]: [좌상, 우상, 우하, 좌하] 순서로 정리된 네 모서리 좌표입니다.
    ordered_quad: Optional[np.ndarray]
    # [1차]: 투시 변환으로 반듯하게 펴진 컬러 문서 이미지(실패 시 None).
    # [2차]: 찌그러진 종이를 정면 직사각형으로 다림질한 컬러 사진입니다.
    warped: Optional[np.ndarray]
    # [1차]: 적응형 이진화가 적용된 최종 흑백 스캔 이미지(실패 시 None).
    # [2차]: 글씨만 또렷하게 인쇄된 최종 스캔 결과물입니다.
    scanned: Optional[np.ndarray]
    # [1차]: 적용된 3×3 투시 변환 행렬(Homography, 실패 시 None).
    # [2차]: 종이를 펼 때 사용한 3×3 수학 공식 행렬입니다.
    transform_matrix: Optional[np.ndarray]
    # [1차]: 검출된 문서(또는 최대 후보)의 픽셀 면적입니다.
    # [2차]: 문서가 차지하는 넓이(픽셀 수)입니다.
    contour_area: float
    # [1차]: 전체 화면 대비 문서 면적 비율입니다.
    # [2차]: 화면에서 종이가 몇 %를 차지하는지 나타냅니다.
    area_ratio: float
    # [1차]: 실패 시 상세 원인 문자열(성공 시 None).
    # [2차]: 왜 실패했는지 적어둔 진단 메모입니다.
    failure_reason: Optional[str]
    # [1차]: 해당 실행에 실제 적용된 ScanConfig 객체입니다.
    # [2차]: 이 스캔에 사용된 다이얼 설정표 사본입니다.
    config: ScanConfig


# [1차]: 전처리 → 윤곽선 → 기하 변환 → 후처리를 일관되게 실행하는 규칙 기반 문서 스캐너 클래스입니다.
# [2차]: 5개 작업 부서를 하나의 자동 컨베이어 벨트로 연결해 돌리는 공장장 클래스입니다.
class DocumentScanner:
    """Rule-based OpenCV + NumPy document detection and rectification pipeline."""

    # [1차]: 기본 설정을 주입받아 스캐너 인스턴스를 초기화합니다.
    # [2차]: 별도 설정을 안 주면 표준 레시피(ScanConfig)를 기본값으로 장착합니다.
    def __init__(self, default_config: Optional[ScanConfig] = None) -> None:
        # [1차]: None이 전달되면 기본 ScanConfig()를 생성해 저장합니다.
        # [2차]: 공장 기본 설정표를 책상 위에 올려둡니다.
        self.default_config = default_config or ScanConfig()

    # [1차]: 메모리 상의 BGR 이미지 배열에 대해 전체 5단계 스캔 파이프라인을 실행합니다.
    # [2차]: 사진 한 장을 컨베이어 벨트 입구에 올리면 5개 공정을 쭉 통과시켜 결과 폴더를 만들어 줍니다.
    def scan_image(
        self,
        image: np.ndarray,
        config: Optional[ScanConfig] = None,
    ) -> ScanResult:
        """Execute full 5-stage pipeline on an in-memory BGR image."""
        # [1차]: 유효한 비어 있지 않은 NumPy 이미지 배열인지 검증합니다.
        # [2차]: 빈 손으로 벨트에 올리면 작업 시작 전에 막아줍니다.
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            # [1차]: 잘못된 입력에 대해 ValueError를 발생시킵니다.
            # [2차]: "제대로 된 사진이 아닙니다!"라고 알립니다.
            raise ValueError("유효한 입력 이미지 배열이 아닙니다.")

        # [1차]: 호출 시 전달된 임시 config가 있으면 우선 사용하고, 없으면 기본 설정을 사용합니다.
        # [2차]: 손님이 특별 주문 레시피를 줬으면 그걸 쓰고, 아니면 기본 레시피로 진행합니다.
        cfg = config or self.default_config
        # [1차]: 1단계 전처리(흑백 → 블러 → Canny 엣지)를 실행합니다.
        # [2차]: 스케치 부서에서 밑그림(엣지)을 따옵니다.
        prep: PreprocessArtifacts = preprocess_image(image, cfg)
        # [1차]: 2단계 윤곽선 탐색 및 4꼭짓점 사각형 검출을 실행합니다.
        # [2차]: 가위질 부서에서 가장 크고 반듯한 네모 종이를 찾습니다.
        contour_res: ContourDetectionResult = find_document_contour(prep.edges, image, cfg)

        # [1차]: 사각형 검출에 실패한 경우 크래시 대신 중간 결과물과 실패 사유를 담아 반환합니다.
        # [2차]: 네모를 못 찾았더라도 멈추지 않고 "여기까지 진행했고 이런 이유로 실패했어요" 보고서를 냅니다.
        if contour_res.quad_points is None:
            # [1차]: detected=False 상태의 ScanResult를 반환합니다.
            # [2차]: 중간 사진들(블러·엣지·오버레이)까지 고스란히 담아 원인 분석이 가능하게 합니다.
            return ScanResult(
                detected=False,
                original=image.copy(),
                grayscale=prep.grayscale,
                blurred=prep.blurred,
                edges=prep.edges,
                contour_overlay=contour_res.overlay_image,
                ordered_quad=None,
                warped=None,
                scanned=None,
                transform_matrix=None,
                contour_area=contour_res.contour_area,
                area_ratio=contour_res.area_ratio,
                failure_reason=contour_res.failure_reason or "document_not_detected",
                config=cfg,
            )

        # [1차]: 3단계 수동 정렬·투시 변환과 4단계 적응형 이진화를 예외 방어 블록 안에서 실행합니다.
        # [2차]: 네 점을 펴고 흑백 인쇄하는 중 혹시라도 기하 오류가 나면 안전하게 잡아냅니다.
        try:
            # [1차]: 수동 4점 정렬 및 정면 직사각형 투시 변환을 수행합니다.
            # [2차]: 기하학 장인이 비스듬한 문서를 반듯하게 다림질합니다.
            warped, ordered_quad, matrix = four_point_transform(image, contour_res.quad_points)
            # [1차]: 펴진 문서에 적응형 이진화를 적용해 깔끔한 흑백 스캔본을 만듭니다.
            # [2차]: 복사기 담당이 그림자를 걷어내고 선명한 흑백 문서로 뽑아냅니다.
            scanned = postprocess_scan(warped, cfg)
        except ValueError as exc:
            # [1차]: 일직선 퇴화 좌표 등 기하학적 예외 발생 시 안전하게 실패 결과를 반환합니다.
            # [2차]: 좌표가 일직선으로 겹치는 예외가 나면 앱이 꺼지지 않게 실패 사유를 적어 반환합니다.
            return ScanResult(
                detected=False,
                original=image.copy(),
                grayscale=prep.grayscale,
                blurred=prep.blurred,
                edges=prep.edges,
                contour_overlay=contour_res.overlay_image,
                ordered_quad=None,
                warped=None,
                scanned=None,
                transform_matrix=None,
                contour_area=contour_res.contour_area,
                area_ratio=contour_res.area_ratio,
                failure_reason=f"geometry_error: {exc}",
                config=cfg,
            )

        # [1차]: 최종 정렬된 [TL, TR, BR, BL] 순서대로 라벨이 정확히 매칭된 오버레이 이미지를 새로 그립니다.
        # [2차]: 정렬이 끝난 진짜 좌상·우상·우하·좌하 위치에 정확한 이름표를 다시 깔끔하게 붙여줍니다.
        ordered_overlay = image.copy()
        # [1차]: 입력이 흑백이면 컬러 시각화를 위해 BGR로 변환합니다.
        # [2차]: 흑백 사진이면 색깔 펜이 보이도록 컬러 도화지로 바꿉니다.
        if ordered_overlay.ndim == 2:
            ordered_overlay = cv2.cvtColor(ordered_overlay, cv2.COLOR_GRAY2BGR)
        # [1차]: 정렬된 4꼭짓점을 정수 픽셀 좌표 (4, 1, 2)로 변환합니다.
        # [2차]: 네 모서리 위치를 정수 픽셀 칸에 맞춥니다.
        pts_int = np.round(ordered_quad).astype(np.int32).reshape((-1, 1, 2))
        # [1차]: 초록색(0, 220, 0) 두께 3의 닫힌 사각형 테두리를 그립니다.
        # [2차]: 확정된 문서 테두리를 초록 형광펜으로 굵게 둘러줍니다.
        cv2.polylines(ordered_overlay, [pts_int], True, (0, 220, 0), 3)
        # [1차]: 정렬 순서와 1:1 일치하는 TL, TR, BR, BL 라벨과 빨간 원을 각 꼭짓점에 표시합니다.
        # [2차]: 네 모서리에 빨간 압정을 꽂고 진짜 자리 이름표(TL/TR/BR/BL)를 붙입니다.
        for label, pt in zip(["TL", "TR", "BR", "BL"], pts_int.reshape(4, 2)):
            # [1차]: 정수 (x, y) 중심 좌표를 만듭니다.
            # [2차]: 압정을 꽂을 정확한 좌표를 꺼냅니다.
            center = (int(pt[0]), int(pt[1]))
            # [1차]: 빨간색 채워진 원을 그립니다.
            # [2차]: 빨간 압정을 찍습니다.
            cv2.circle(ordered_overlay, center, 8, (0, 0, 255), -1)
            # [1차]: 노란색 글씨로 방위 라벨을 표시합니다.
            # [2차]: 옆에 노란색으로 TL/TR/BR/BL 글씨를 써줍니다.
            cv2.putText(
                ordered_overlay,
                label,
                (center[0] + 10, max(22, center[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # [1차]: 모든 단계가 성공적으로 완료된 최종 ScanResult 객체를 반환합니다.
        # [2차]: 5개 공정을 완벽히 통과한 최종 결과 폴더를 손님에게 전달합니다.
        return ScanResult(
            detected=True,
            original=image.copy(),
            grayscale=prep.grayscale,
            blurred=prep.blurred,
            edges=prep.edges,
            contour_overlay=ordered_overlay,
            ordered_quad=ordered_quad,
            warped=warped,
            scanned=scanned,
            transform_matrix=matrix,
            contour_area=contour_res.contour_area,
            area_ratio=contour_res.area_ratio,
            failure_reason=None,
            config=cfg,
        )

    # [1차]: 파일 경로에서 한글 안전 로드 후 전체 스캔 파이프라인을 실행하는 편의 메서드입니다.
    # [2차]: 사진 파일 주소만 알려주면 알아서 꺼내와 스캔까지 한 번에 끝내줍니다.
    def scan_file(
        self,
        image_path: PathLike,
        config: Optional[ScanConfig] = None,
    ) -> ScanResult:
        """Load an image file safely and run the full scanning pipeline."""
        # [1차]: 한글/공백 안전 로더로 이미지를 디코딩합니다.
        # [2차]: 다국어 택배 기사가 사진 파일을 안전하게 열어옵니다.
        image = load_image(image_path)
        # [1차]: 로드한 이미지를 scan_image로 전달해 최종 결과를 반환합니다.
        # [2차]: 열어온 사진을 컨베이어 벨트에 올려 스캔 결과를 받아옵니다.
        return self.scan_image(image, config=config)


# [1차]: 다양한 해상도·채널의 단계별 이미지를 동일한 크기의 제목표시줄 포함 BGR 타일로 규격화합니다.
# [2차]: 크기가 제각각인 사진들을 똑같은 크기의 액자 틀에 비율이 깨지지 않게 예쁘게 끼워 넣습니다.
def _prepare_tile(
    image: Optional[np.ndarray],
    title: str,
    tile_size: Tuple[int, int],
) -> np.ndarray:
    """Resize any grayscale/color image into a uniform labeled BGR tile."""
    # [1차]: 목표 타일의 가로·세로 크기와 상단 제목 바 높이(32px)를 설정합니다.
    # [2차]: 액자 가로·세로 크기와 위에 붙일 이름표 높이를 정합니다.
    tile_w, tile_h = tile_size
    # [1차]: 제목 표시줄 높이를 32픽셀로 고정합니다.
    # [2차]: 위쪽 제목 띠를 32픽셀 두께로 잡습니다.
    header_h = 32
    # [1차]: 어두운 배경색(RGB 24)으로 채워진 빈 액자 캔버스를 생성합니다.
    # [2차]: 진회색 배경의 빈 액자 판을 준비합니다.
    canvas = np.full((tile_h + header_h, tile_w, 3), 24, dtype=np.uint8)

    # [1차]: 이미지가 None(검출 실패 단계)이면 중앙에 빨간색 NOT DETECTED 안내 문구를 표시합니다.
    # [2차]: 실패해서 결과 사진이 비어 있으면 액자 한가운데 빨간 글씨로 "미검출"이라고 적어줍니다.
    if image is None or image.size == 0:
        # [1차]: 캔버스 중앙에 경고 문구를 렌더링합니다.
        # [2차]: 빨간 글씨로 "NOT DETECTED" 도장을 찍습니다.
        cv2.putText(
            canvas,
            "NOT DETECTED",
            (max(10, tile_w // 2 - 90), header_h + tile_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        # [1차]: 2차원 흑백 이미지이면 3채널 BGR로 변환해 컬러 캔버스에 합성할 수 있게 합니다.
        # [2차]: 흑백 사진도 컬러 액자에 딱 맞게 3채널로 모양을 통일합니다.
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        # [1차]: 원본 높이와 너비를 추출합니다.
        # [2차]: 사진의 원래 세로·가로 크기를 확인합니다.
        h, w = bgr.shape[:2]
        # [1차]: 가로세로 종횡비를 유지하면서 타일 안에 쏙 들어가도록 축소 비율을 계산합니다.
        # [2차]: 사진이 찌그러지지 않게 가로·세로 중 더 빡빡한 쪽 기준으로 똑같이 줄입니다.
        scale = min(tile_w / max(w, 1), tile_h / max(h, 1))
        # [1차]: 비율이 적용된 새 가로 픽셀 크기를 1 이상 정수로 계산합니다.
        # [2차]: 줄어든 사진의 가로 길이를 구합니다.
        new_w = max(1, int(round(w * scale)))
        # [1차]: 비율이 적용된 새 세로 픽셀 크기를 1 이상 정수로 계산합니다.
        # [2차]: 줄어든 사진의 세로 길이를 구합니다.
        new_h = max(1, int(round(h * scale)))
        # [1차]: 축소 시 계단 현상을 줄이는 INTER_AREA 보간법으로 이미지를 리사이즈합니다.
        # [2차]: 사진을 깨짐 없이 부드럽게 축소합니다.
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # [1차]: 타일 중앙 정렬을 위한 좌우 여백(offset_x)을 계산합니다.
        # [2차]: 남는 좌우 공간의 절반만큼 밀어 정중앙에 맞춥니다.
        offset_x = (tile_w - new_w) // 2
        # [1차]: 제목 바 아래 중앙 정렬을 위한 상하 여백(offset_y)을 계산합니다.
        # [2차]: 제목 띠 아래 정중앙 높이에 맞춥니다.
        offset_y = header_h + (tile_h - new_h) // 2
        # [1차]: NumPy 슬라이싱으로 캔버스 중앙 영역에 리사이즈된 이미지를 복사합니다.
        # [2차]: 빈 액자 정중앙에 사진을 딱 붙여 넣습니다.
        canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

    # [1차]: 상단 제목 표시줄 배경 사각형을 채웁니다.
    # [2차]: 액자 맨 위에 차분한 남색 제목 띠를 두릅니다.
    cv2.rectangle(canvas, (0, 0), (tile_w, header_h), (45, 52, 64), -1)
    # [1차]: 제목 텍스트를 밝은 회백색으로 출력합니다.
    # [2차]: 제목 띠 위에 단계 이름(예: "3. Canny Edges")을 깔끔하게 적습니다.
    cv2.putText(
        canvas,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 244, 248),
        1,
        cv2.LINE_AA,
    )
    # [1차]: 타일 전체 외곽에 1픽셀 회색 테두리를 그려 인접 타일과 구분합니다.
    # [2차]: 액자 겉면에 얇은 테두리 선을 둘러 옆 액자와 깔끔하게 경계를 나눕니다.
    cv2.rectangle(canvas, (0, 0), (tile_w - 1, tile_h + header_h - 1), (90, 100, 115), 1)
    # [1차]: 완성된 단일 타일 이미지를 반환합니다.
    # [2차]: 완성된 액자 1개를 돌려줍니다.
    return canvas


# [1차]: 6개 처리 단계(원본·블러·엣지·윤곽선·워핑·최종 스캔)를 2행×3열 한 장의 몽타주 이미지로 합칩니다.
# [2차]: 6단계 변신 과정을 한눈에 볼 수 있는 6컷 만화 포스터로 만들어 줍니다.
def build_stage_montage(
    result: ScanResult,
    tile_size: Tuple[int, int] = (360, 270),
) -> np.ndarray:
    """Build a 2x3 visual overview showing all pipeline stages side-by-side."""
    # [1차]: 6개 단계별 이미지를 제목과 함께 동일 규격 타일 리스트로 생성합니다.
    # [2차]: 1번 원본부터 6번 최종 스캔까지 액자 6개를 차례대로 만듭니다.
    tiles = [
        _prepare_tile(result.original, "1. Original", tile_size),
        _prepare_tile(result.blurred, f"2. Gray + Blur (k={result.config.blur_kernel})", tile_size),
        _prepare_tile(
            result.edges,
            f"3. Canny Edges ({result.config.canny_low}, {result.config.canny_high})",
            tile_size,
        ),
        _prepare_tile(result.contour_overlay, "4. Contour & 4 Corners", tile_size),
        _prepare_tile(result.warped, "5. Perspective Warped", tile_size),
        _prepare_tile(result.scanned, "6. Adaptive Threshold Scan", tile_size),
    ]
    # [1차]: 앞 3개 타일을 가로 방향(hstack)으로 이어 1행을 구성합니다.
    # [2차]: 1~3번 액자를 윗줄에 나란히 붙입니다.
    top_row = np.hstack(tiles[:3])
    # [1차]: 뒤 3개 타일을 가로 방향(hstack)으로 이어 2행을 구성합니다.
    # [2차]: 4~6번 액자를 아랫줄에 나란히 붙입니다.
    bottom_row = np.hstack(tiles[3:])
    # [1차]: 1행과 2행을 세로 방향(vstack)으로 쌓아 완성된 2×3 몽타주를 반환합니다.
    # [2차]: 윗줄과 아랫줄을 위아래로 합쳐 6컷 변신 포스터를 완성합니다.
    return np.vstack([top_row, bottom_row])


# [1차]: 6패널 몽타주와 각 핵심 단계 이미지를 지정 폴더에 일괄 저장하고 경로 사전을 반환합니다.
# [2차]: GUI 창을 띄울 수 없는 환경(Colab 등)에서도 눈으로 확인하도록 모든 단계 사진을 폴더에 저장합니다.
def save_stage_outputs(
    result: ScanResult,
    output_dir: PathLike,
    stem: str,
) -> Dict[str, Path]:
    """Save final scan image, 6-panel montage, and intermediate stage images to disk."""
    # [1차]: 출력 폴더 Path 객체를 만들고 존재하지 않으면 생성합니다.
    # [2차]: 결과물 앨범을 보관할 폴더를 먼저 준비합니다.
    out_root = Path(output_dir)
    # [1차]: 중첩 폴더 포함 생성(parents=True, exist_ok=True)을 수행합니다.
    # [2차]: 폴더가 없어도 알아서 새로 만듭니다.
    out_root.mkdir(parents=True, exist_ok=True)
    # [1차]: 저장된 단계명 → 파일 경로 매핑 딕셔너리를 초기화합니다.
    # [2차]: "어떤 사진을 어디에 저장했는지" 영수증 목록을 준비합니다.
    saved: Dict[str, Path] = {}

    # [1차]: 6패널 요약 몽타주를 생성합니다.
    # [2차]: 6컷 변신 포스터를 그립니다.
    montage = build_stage_montage(result)
    # [1차]: 몽타주·엣지·윤곽선 이미지를 한글 안전 save_image로 저장합니다.
    # [2차]: 포스터와 엣지·윤곽선 사진을 폴더에 차곡차곡 저장합니다.
    saved["montage"] = save_image(out_root / f"{stem}_stages_montage.png", montage)
    saved["edges"] = save_image(out_root / f"{stem}_edges.png", result.edges)
    saved["contour"] = save_image(out_root / f"{stem}_contour.png", result.contour_overlay)

    # [1차]: 문서가 정상 검출되어 워핑 이미지가 존재하면 저장합니다.
    # [2차]: 다림질된 컬러 종이 사진이 있으면 함께 보관합니다.
    if result.warped is not None:
        saved["warped"] = save_image(out_root / f"{stem}_warped.png", result.warped)
    # [1차]: 최종 적응형 이진화 스캔본이 존재하면 저장합니다.
    # [2차]: 최종 흑백 스캔 문서가 있으면 함께 보관합니다.
    if result.scanned is not None:
        saved["scanned"] = save_image(out_root / f"{stem}_scanned.png", result.scanned)
    # [1차]: 저장된 모든 파일 경로 사전을 반환합니다.
    # [2차]: 저장 완료 영수증 묶음을 돌려줍니다.
    return saved
