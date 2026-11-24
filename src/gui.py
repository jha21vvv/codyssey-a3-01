"""Interactive OpenCV GUI with real-time trackbars, step windows, and webcam mode."""

# [1차]: 타입 어노테이션 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 나중에 해석하도록 미리 켜둡니다.
from __future__ import annotations

# [1차]: 웹캠 캡처 파일명 충돌을 막는 타임스탬프 생성을 위해 datetime을 가져옵니다.
# [2차]: 웹캠으로 찍은 사진마다 "몇 시 몇 분 몇 초" 시간 이름표를 붙이기 위해 시계를 꺼냅니다.
from datetime import datetime

# [1차]: 파일 경로 생성을 위해 pathlib.Path를 가져옵니다.
# [2차]: 저장 주소를 안전하게 다루는 스마트 주소 객체입니다.
from pathlib import Path

# [1차]: 리스트 및 선택적 타입 힌트를 가져옵니다.
# [2차]: 반환 목록과 선택 인자를 표시하는 라벨입니다.
from typing import List, Optional

# [1차]: OpenCV 윈도우, 트랙바, 키 입력 대기, 웹캠 캡처를 위해 cv2를 가져옵니다.
# [2차]: 실시간 화면 창과 다이얼 슬라이더, 카메라를 제어하는 스튜디오 장비입니다.
import cv2

# [1차]: 미검출 안내 빈 캔버스 생성을 위해 NumPy를 가져옵니다.
# [2차]: 문서를 못 찾았을 때 띄울 안내판 도화지를 만들 때 씁니다.
import numpy as np

# [1차]: 설정 객체와 홀수 커널 보정 함수를 가져옵니다.
# [2차]: 트랙바 숫자를 안전한 홀수로 다듬는 보정기와 설정표를 불러옵니다.
from .config import ScanConfig, normalize_odd_kernel

# [1차]: 한글 안전 이미지 입출력 유틸리티를 가져옵니다.
# [2차]: 다국어 택배 기사를 불러옵니다.
from .io_utils import PathLike, load_image, save_image

# [1차]: 문서 스캐너 엔진과 몽타주 생성기를 가져옵니다.
# [2차]: 실시간으로 사진을 돌려줄 컨베이어 벨트 공장장을 불러옵니다.
from .pipeline import DocumentScanner, ScanResult, build_stage_montage


# [1차]: cv2.createTrackbar가 필수 인자로 요구하는 빈 콜백 함수입니다(실제 값은 루프에서 폴링).
# [2차]: 슬라이더가 움직일 때 호출되는 빈 벨 버튼입니다(어차피 매 순간 눈금을 직접 읽습니다).
def _noop(_value: int) -> None:
    """No-op callback required by cv2.createTrackbar."""


# [1차]: 고해상도 이미지가 모니터를 벗어나지 않도록 가로 max_width 이하로 비율 축소합니다.
# [2차]: 너무 큰 사진이 화면 밖으로 삐져나가지 않게 모니터 크기에 맞춰 알맞게 줄여줍니다.
def _resize_for_window(image: np.ndarray, max_width: int = 520) -> np.ndarray:
    """Downscale large images proportionally for comfortable desktop display."""
    # [1차]: 현재 이미지의 높이와 너비를 추출합니다.
    # [2차]: 사진 크기가 얼마나 큰지 재봅니다.
    h, w = image.shape[:2]
    # [1차]: 이미 max_width 이하이면 리사이즈 없이 원본 그대로 반환합니다.
    # [2차]: 이미 화면에 쏙 들어오는 크기라면 그대로 보여줍니다.
    if w <= max_width:
        return image
    # [1차]: 가로 기준 축소 배율을 계산합니다.
    # [2차]: 몇 %로 줄여야 가로 520픽셀에 딱 맞는지 비율을 구합니다.
    scale = max_width / float(w)
    # [1차]: 종횡비를 유지한 채 INTER_AREA로 선명하게 축소하여 반환합니다.
    # [2차]: 찌그러짐 없이 깔끔하게 줄인 화면용 사진을 돌려줍니다.
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


# [1차]: 5단계 OpenCV 윈도우와 실시간 Blur/Canny/Morph 트랙바를 띄워 인터랙티브 스캔을 실행합니다.
# [2차]: 5개 모니터 창을 나란히 띄워두고 슬라이더를 밀 때마다 엣지와 스캔 결과가 즉시 바뀌는 체험 스튜디오입니다.
def run_interactive_gui(
    image_path: PathLike,
    initial_config: Optional[ScanConfig] = None,
    output_path: Optional[PathLike] = None,
    max_iterations: Optional[int] = None,
) -> ScanResult:
    """Launch multi-window pipeline viewer with live blur and Canny trackbars.

    Keys:
      - 's': Save current scanned result (and stage montage)
      - 'q' or ESC: Exit GUI cleanly
    """
    # [1차]: 입력 이미지 파일을 한글 안전 로더로 읽어옵니다.
    # [2차]: 먼저 스캔할 사진을 책상 위로 불러옵니다.
    image = load_image(image_path)
    # [1차]: 초기 설정이 없으면 기본 ScanConfig를 사용합니다.
    # [2차]: 시작 다이얼 눈금을 준비합니다.
    cfg = initial_config or ScanConfig()
    # [1차]: 문서 스캐너 인스턴스를 생성합니다.
    # [2차]: 컨베이어 벨트 기계를 켭니다.
    scanner = DocumentScanner(cfg)

    # [1차]: 트랙바 컨트롤 창 제목을 정의합니다.
    # [2차]: 슬라이더 조종판 창의 이름표입니다.
    ctrl_win = "Document Scanner Controls"
    # [1차]: 원본 → 전처리 → 엣지 → 윤곽선 → 스캔 결과 5단계 창 제목 매핑입니다.
    # [2차]: 5단계 공정 모니터 각각에 붙일 이름표 5개입니다.
    stage_windows = {
        "orig": "1. Original",
        "blur": "2. Preprocess (Blur)",
        "edge": "3. Canny Edges",
        "contour": "4. Document Contour",
        "scan": "5. Scanned Result",
    }

    # [1차]: GUI 미지원(Colab/헤드리스) 환경에서 발생하는 cv2.error를 잡아 친절한 안내 예외로 전환합니다.
    # [2차]: 모니터 창을 못 띄우는 서버 환경이라면 멈추지 않고 CLI 대안 명령어를 친절히 알려줍니다.
    try:
        # [1차]: 크기 조절 가능한 컨트롤 창을 생성하고 크기를 460×220으로 맞춥니다.
        # [2차]: 조종판 창을 화면에 엽니다.
        cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ctrl_win, 460, 220)
        # [1차]: 5개 단계별 시각화 윈도우를 모두 생성합니다.
        # [2차]: 5개 모니터 창을 나란히 켭니다.
        for win_name in stage_windows.values():
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # [1차]: 블러 커널(0~31), Canny 하한/상한(0~255), 모폴로지 닫힘(0~15) 실시간 트랙바 4개를 부착합니다.
        # [2차]: 마우스로 드래그할 수 있는 실시간 조절 슬라이더 4개를 조종판에 달아줍니다.
        cv2.createTrackbar("Blur Kernel", ctrl_win, cfg.blur_kernel, 31, _noop)
        cv2.createTrackbar("Canny Low", ctrl_win, cfg.canny_low, 255, _noop)
        cv2.createTrackbar("Canny High", ctrl_win, cfg.canny_high, 255, _noop)
        cv2.createTrackbar("Morph Close", ctrl_win, cfg.morph_kernel, 15, _noop)
    except cv2.error as exc:
        # [1차]: GUI 초기화 실패 시 헤드리스 CLI 명령어를 포함한 RuntimeError를 발생시킵니다.
        # [2차]: "여긴 화면 창을 못 여니 대신 CLI 저장 명령어를 써보세요!"라고 안내합니다.
        raise RuntimeError(
            "OpenCV GUI 윈도우를 초기화할 수 없습니다(헤드리스/Colab 환경). "
            "'python main.py scan <이미지경로> --save-stages' 명령을 사용하세요."
        ) from exc

    # [1차]: 초기 설정으로 1회 스캔 결과를 준비하고 반복 횟수 카운터를 0으로 초기화합니다.
    # [2차]: 첫 화면에 띄울 초기 결과물을 미리 한 번 뽑아둡니다.
    last_result = scanner.scan_image(image, cfg)
    iteration = 0

    # [1차]: 어떤 종료 경로에서도 cv2.destroyAllWindows()가 반드시 호출되도록 try-finally로 감쌉니다.
    # [2차]: 프로그램이 끝나거나 에러가 나도 열려 있던 창들을 깔끔히 닫고 퇴근하게 보장합니다.
    try:
        # [1차]: 사용자가 'q'/ESC를 누르거나 max_iterations에 도달할 때까지 실시간 갱신 루프를 돕니다.
        # [2차]: 종료 키를 누를 때까지 슬라이더 값을 계속 읽으며 5개 화면을 실시간으로 새로고침합니다.
        while True:
            # [1차]: 각 트랙바의 현재 위치를 읽어 안전한 홀수/범위 값으로 수집합니다.
            # [2차]: 사용자가 마우스로 맞춘 슬라이더 눈금 4개를 실시간으로 읽어옵니다.
            blur_k = normalize_odd_kernel(cv2.getTrackbarPos("Blur Kernel", ctrl_win), 1, 31)
            canny_lo = cv2.getTrackbarPos("Canny Low", ctrl_win)
            canny_hi = cv2.getTrackbarPos("Canny High", ctrl_win)
            morph_k = cv2.getTrackbarPos("Morph Close", ctrl_win)

            # [1차]: 실시간 트랙바 값으로 새 ScanConfig를 구성합니다(low>high 역전 자동 방어).
            # [2차]: 방금 읽은 슬라이더 눈금으로 새 레시피 카드를 만듭니다.
            live_cfg = ScanConfig(
                blur_kernel=blur_k,
                canny_low=canny_lo,
                canny_high=canny_hi,
                morph_kernel=morph_k,
                min_area_ratio=cfg.min_area_ratio,
                max_area_ratio=cfg.max_area_ratio,
                approx_epsilon_ratio=cfg.approx_epsilon_ratio,
                adaptive_block_size=cfg.adaptive_block_size,
                adaptive_c=cfg.adaptive_c,
                use_clahe=cfg.use_clahe,
            )
            # [1차]: 새 설정으로 전체 파이프라인을 재실행합니다.
            # [2차]: 바뀐 레시피로 즉시 사진을 다시 처리합니다.
            last_result = scanner.scan_image(image, live_cfg)

            # [1차]: 1~4단계(원본, 블러, 엣지, 윤곽선 오버레이)를 각 전용 창에 표시합니다.
            # [2차]: 1번부터 4번 모니터 화면을 새 결과로 업데이트합니다.
            cv2.imshow(stage_windows["orig"], _resize_for_window(last_result.original))
            cv2.imshow(stage_windows["blur"], _resize_for_window(last_result.blurred))
            cv2.imshow(stage_windows["edge"], _resize_for_window(last_result.edges))
            cv2.imshow(stage_windows["contour"], _resize_for_window(last_result.contour_overlay))

            # [1차]: 5단계 창에는 스캔 성공 시 이진화 결과를, 실패 시 트랙바 조절 안내판을 띄웁니다.
            # [2차]: 문서를 찾았으면 최종 스캔본을, 못 찾았으면 "슬라이더를 조절해 보세요" 안내판을 보여줍니다.
            if last_result.scanned is not None:
                cv2.imshow(stage_windows["scan"], _resize_for_window(last_result.scanned))
            else:
                # [1차]: 어두운 회색 안내 캔버스(300×420)를 생성합니다.
                # [2차]: 검은 칠판을 하나 준비합니다.
                blank = np.full((300, 420, 3), 30, dtype=np.uint8)
                # [1차]: 미검출 상태 메시지를 빨간 글씨로 적습니다.
                # [2차]: "문서를 못 찾았습니다"라고 칠판에 씁니다.
                cv2.putText(
                    blank,
                    "Document Not Detected",
                    (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (70, 90, 255),
                    2,
                    cv2.LINE_AA,
                )
                # [1차]: 트랙바 조정 가이드를 흰색 글씨로 적습니다.
                # [2차]: "블러/Canny 트랙바를 움직여 보세요"라고 힌트를 씁니다.
                cv2.putText(
                    blank,
                    "Adjust Trackbars (Blur / Canny)",
                    (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                # [1차]: 안내 캔버스를 5단계 창에 출력합니다.
                # [2차]: 칠판을 5번 모니터에 띄웁니다.
                cv2.imshow(stage_windows["scan"], blank)

            # [1차]: 30ms 동안 키 입력을 대기하고 하위 8비트 ASCII 코드만 추출합니다.
            # [2차]: 사용자가 키보드에서 's'나 'q'를 눌렀는지 잠깐 귀 기울여 확인합니다.
            key = cv2.waitKey(30) & 0xFF
            # [1차]: 's' 키 입력 시 현재 스캔 결과와 6패널 몽타주를 디스크에 저장합니다.
            # [2차]: 's'(Save)를 누르면 마음에 드는 지금 결과물과 6컷 포스터를 파일로 저장합니다.
            if key == ord("s"):
                # [1차]: 지정 출력 경로가 없으면 outputs/gui_scan_<원본명>.png를 기본값으로 사용합니다.
                # [2차]: 저장할 파일 이름을 정합니다.
                target = Path(output_path) if output_path else Path("outputs") / f"gui_scan_{Path(image_path).stem}.png"
                # [1차]: 최종 스캔본이 있으면 저장합니다.
                # [2차]: 흑백 스캔 문서를 저장합니다.
                if last_result.scanned is not None:
                    save_image(target, last_result.scanned)
                # [1차]: 단계별 6패널 몽타주도 나란히 저장합니다.
                # [2차]: 6컷 변신 포스터도 같이 저장해 둡니다.
                save_image(target.with_name(f"{target.stem}_montage.png"), build_stage_montage(last_result))
            # [1차]: 'q' 또는 ESC(27) 키 입력 시 루프를 정상 종료합니다.
            # [2차]: 'q'(Quit)나 ESC를 누르면 촬영 스튜디오를 마감합니다.
            elif key in (ord("q"), 27):
                break

            # [1차]: 테스트용 반복 제한(max_iterations) 도달 시 안전하게 루프를 빠져나옵니다.
            # [2차]: 자동 테스트에서는 무한 대기하지 않도록 정해진 횟수만큼만 돌고 나옵니다.
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
    finally:
        # [1차]: 열려 있는 모든 OpenCV GUI 창 리소스를 해제합니다.
        # [2차]: 작업이 끝나면 모든 모니터 창을 깔끔하게 끕니다.
        cv2.destroyAllWindows()

    # [1차]: 종료 직전의 마지막 ScanResult를 반환합니다.
    # [2차]: 마지막으로 조절했던 최종 결과물을 돌려줍니다.
    return last_result


# [1차]: 실시간 웹캠 스트림에서 문서를 검출하고 's'로 캡처 저장, 'q'로 안전 종료하는 웹캠 모드입니다.
# [2차]: 노트북/USB 카메라로 종이를 비추면 초록 네모를 실시간으로 따라 그리다 's'를 누르면 찰칵 스캔해 주는 모드입니다.
def run_webcam_mode(
    camera_index: int = 0,
    config: Optional[ScanConfig] = None,
    output_dir: PathLike = "outputs/webcam",
    max_frames: Optional[int] = None,
) -> List[Path]:
    """Run real-time webcam document scanner ('s' saves warped scan, 'q' exits)."""
    # [1차]: 설정과 스캐너 객체, 저장 파일 경로 기록 리스트를 초기화합니다.
    # [2차]: 카메라 스캔 설정과 "저장한 사진 영수증 목록"을 준비합니다.
    cfg = config or ScanConfig()
    scanner = DocumentScanner(cfg)
    saved_files: List[Path] = []

    # [1차]: 지정 인덱스의 카메라 장치를 오픈합니다.
    # [2차]: 0번(기본 내장/USB) 웹캠 전원을 켭니다.
    cap = cv2.VideoCapture(int(camera_index))
    # [1차]: 카메라 연결 실패 또는 OS 권한 차단 시 리소스를 정리하고 명확한 안내 예외를 발생시킵니다.
    # [2차]: 웹캠이 안 꽂혀 있거나 권한이 막혀 있으면 바로 안내하고 멈춥니다.
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"웹캠(인덱스 {camera_index})을 열 수 없습니다. "
            "카메라 연결 상태 및 OS 카메라 권한을 확인하세요."
        )

    # [1차]: 실시간 표시 창 제목과 프레임 카운터를 초기화합니다.
    # [2차]: 카메라 모니터 창 이름표를 정합니다.
    win_name = "Webcam Document Scanner [s: Save | q: Quit]"
    frame_idx = 0

    # [1차]: 종료 시 카메라 핸들(cap.release)과 GUI 창을 반드시 해제하도록 try-finally로 감쌉니다.
    # [2차]: 어떤 일이 생겨도 웹캠 불빛이 계속 켜져 있지 않도록 안전하게 끄고 정리합니다.
    try:
        # [1차]: 실시간 프레임을 반복 수신하며 문서를 검출·시각화합니다.
        # [2차]: 카메라가 보내오는 장면을 1초에 수십 장씩 실시간으로 스캔합니다.
        while True:
            # [1차]: 카메라에서 다음 프레임 1장을 읽어옵니다.
            # [2차]: 지금 카메라 렌즈에 비친 장면을 한 컷 받아옵니다.
            ok, frame = cap.read()
            # [1차]: 프레임 수신 실패(장치 분리 등) 시 루프를 안전 종료합니다.
            # [2차]: 카메라 신호가 끊기면 안전하게 종료합니다.
            if not ok or frame is None:
                break

            # [1차]: 현재 프레임에 전체 문서 스캔 파이프라인을 적용합니다.
            # [2차]: 방금 찍힌 화면에서 종이 네모가 있는지 찾아봅니다.
            result = scanner.scan_image(frame, cfg)
            # [1차]: 윤곽선이 그려진 오버레이 사본에 상태 안내 바(HUD)를 덧그립니다.
            # [2차]: 초록 네모가 그려진 화면 맨 위에 상태 안내판을 얹습니다.
            display = result.contour_overlay.copy()
            # [1차]: 검출 여부에 따라 안내 문구와 색상(초록/주황)을 동적으로 선택합니다.
            # [2차]: 종이를 잡았으면 초록색으로 "저장 가능!", 찾는 중이면 주황색으로 "탐색 중..." 표시합니다.
            status_text = "DETECTED - Press 's' to Save, 'q' to Quit" if result.detected else "SEARCHING... Press 'q' to Quit"
            status_color = (0, 230, 0) if result.detected else (0, 140, 255)
            # [1차]: 상단 36픽셀 높이의 어두운 배경 띠를 그립니다.
            # [2차]: 글씨가 잘 보이도록 화면 상단에 검은 띠를 두릅니다.
            cv2.rectangle(display, (0, 0), (display.shape[1], 36), (25, 25, 25), -1)
            # [1차]: 상단 띠 위에 상태 문구를 출력합니다.
            # [2차]: 안내 글자를 띠 위에 적습니다.
            cv2.putText(
                display,
                status_text,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
                cv2.LINE_AA,
            )
            # [1차]: 실시간 결과 프레임을 윈도우에 표시합니다.
            # [2차]: 완성된 라이브 화면을 모니터에 보여줍니다.
            cv2.imshow(win_name, display)

            # [1차]: 1ms 키 입력을 확인합니다.
            # [2차]: 's'나 'q' 키가 눌렸는지 바로 확인합니다.
            key = cv2.waitKey(1) & 0xFF
            # [1차]: 's' 입력 시 현재 스캔본(미검출 시 원본 프레임)을 타임스탬프 파일명으로 저장합니다.
            # [2차]: 's'를 누르면 찰칵! 현재 스캔된 문서를 날짜·시간 이름으로 폴더에 저장합니다.
            if key == ord("s"):
                # [1차]: 마이크로초 단위 타임스탬프로 파일명 중복을 방지합니다.
                # [2차]: 연타해서 찍어도 파일이 덮어쓰이지 않게 1/100만 초 단위 이름표를 붙입니다.
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # [1차]: 출력 디렉터리 Path를 준비합니다.
                # [2차]: 저장 폴더 주소를 준비합니다.
                out_root = Path(output_dir)
                # [1차]: 문서 검출 성공 시 이진화 스캔본을, 실패 시 원본 프레임을 저장 대상으로 선택합니다.
                # [2차]: 스캔이 됐으면 깔끔한 스캔본을, 아직 못 잡았으면 현재 카메라 장면을 저장합니다.
                image_to_save = result.scanned if result.scanned is not None else frame
                # [1차]: 한글 안전 로더/세이버로 PNG 파일을 기록하고 목록에 추가합니다.
                # [2차]: 저장한 뒤 영수증 목록에 파일 주소를 기록합니다.
                saved_path = save_image(out_root / f"webcam_scan_{stamp}.png", image_to_save)
                saved_files.append(saved_path)
            # [1차]: 'q' 또는 ESC(27) 입력 시 루프를 정상 종료합니다.
            # [2차]: 'q'를 누르면 카메라 모드를 깔끔하게 종료합니다.
            elif key in (ord("q"), 27):
                break

            # [1차]: 단위 테스트용 max_frames 제한을 확인합니다.
            # [2차]: 테스트 모드에서는 정해진 프레임 수만 확인하고 빠져나옵니다.
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        # [1차]: 카메라 장치를 OS에 반환하고 모든 OpenCV 창을 닫습니다.
        # [2차]: 웹캠 전원을 끄고 창을 전부 닫아 뒷정리를 마무리합니다.
        cap.release()
        cv2.destroyAllWindows()

    # [1차]: 이번 세션에서 저장된 모든 캡처 파일 경로 리스트를 반환합니다.
    # [2차]: 오늘 웹캠으로 찍어 보관한 파일 영수증 목록을 돌려줍니다.
    return saved_files
