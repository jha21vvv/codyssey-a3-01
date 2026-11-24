"""Command-line interface for the OpenCV Document Scanner."""

# [1차]: 타입 어노테이션 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 실행 시점까지 잠깐 보류해 둡니다.
from __future__ import annotations

# [1차]: 서브커맨드와 옵션 인자를 체계적으로 파싱하기 위해 argparse를 가져옵니다.
# [2차]: 터미널에서 사용자가 친 명령어를 깔끔하게 해석하는 안내 데스크 도구입니다.
import argparse

# [1차]: CLI 실행 결과를 기계·사람 모두 읽기 좋은 JSON으로 출력하기 위해 json을 가져옵니다.
# [2차]: 처리 결과를 깔끔한 영수증(JSON)으로 터미널에 찍어주는 프린터입니다.
import json

# [1차]: 명령줄 인자(sys.argv)와 표준 에러(sys.stderr) 접근을 위해 sys를 가져옵니다.
# [2차]: 터미널 입력창과 에러 방송 스피커를 연결합니다.
import sys

# [1차]: 출력 경로 조립과 파일명 추출(stem)을 위해 Path를 가져옵니다.
# [2차]: 파일 이름과 주소를 다루는 스마트 주소 객체입니다.
from pathlib import Path

# [1차]: 리스트 및 선택적 타입 힌트를 가져옵니다.
# [2차]: 인자 목록 타입을 표시하는 안내 라벨입니다.
from typing import List, Optional

# [1차]: 프리셋 목록과 ScanConfig 설정 객체를 가져옵니다.
# [2차]: 추천 레시피 메뉴판과 설정 카드를 불러옵니다.
from .config import PRESETS, ScanConfig

# [1차]: 20장 밝기 히스토그램 EDA 실행 함수를 가져옵니다.
# [2차]: 사진 건강검진 센터 기능을 불러옵니다.
from .eda import run_dataset_eda

# [1차]: 20장 정량 평가 및 파라미터 스윕 실행 함수를 가져옵니다.
# [2차]: 시험 채점관과 파라미터 딜레마 실험기를 불러옵니다.
from .evaluator import evaluate_dataset, run_parameter_sweep

# [1차]: 실시간 트랙바 GUI와 웹캠 모드 실행 함수를 가져옵니다.
# [2차]: 현장 스튜디오 화면 및 카메라 기능을 불러옵니다.
from .gui import run_interactive_gui, run_webcam_mode

# [1차]: UTF-8 콘솔 설정 및 한글 안전 이미지 저장 함수를 가져옵니다.
# [2차]: 터미널 한글 보호 장치와 다국어 택배 기사를 불러옵니다.
from .io_utils import configure_utf8_console, save_image

# [1차]: 문서 스캐너 엔진과 단계별 출력 저장 함수를 가져옵니다.
# [2차]: 스캔 컨베이어 벨트 공장장과 6컷 포스터 저장기를 불러옵니다.
from .pipeline import DocumentScanner, save_stage_outputs


# [1차]: 선택된 프리셋을 기본값으로 삼고 사용자가 명시한 CLI 옵션만 덮어써 ScanConfig를 생성합니다.
# [2차]: 세트 메뉴(프리셋)를 기본으로 고른 뒤, 손님이 따로 주문한 옵션(예: 블러=9)만 맞춤 변경해 줍니다.
def _build_config_from_args(args: argparse.Namespace) -> ScanConfig:
    """Create a ScanConfig from preset + CLI overrides."""
    # [1차]: 인자에서 프리셋 이름을 안전하게 조회해 기본 ScanConfig를 가져옵니다.
    # [2차]: 선택한 세트 메뉴 레시피를 먼저 펼칩니다.
    preset_name = getattr(args, "preset", None)
    base = PRESETS[preset_name] if preset_name in PRESETS else ScanConfig()
    # [1차]: 개별 CLI 옵션이 None이 아니면 해당 값으로 덮어써 새 ScanConfig를 반환합니다.
    # [2차]: 손님이 별도 숫자를 입력한 항목만 갈아 끼워 최종 설정 카드를 만듭니다.
    return ScanConfig(
        blur_kernel=args.blur if getattr(args, "blur", None) is not None else base.blur_kernel,
        canny_low=args.canny_low if getattr(args, "canny_low", None) is not None else base.canny_low,
        canny_high=args.canny_high if getattr(args, "canny_high", None) is not None else base.canny_high,
        morph_kernel=args.morph if getattr(args, "morph", None) is not None else base.morph_kernel,
        use_clahe=True if getattr(args, "clahe", False) else base.use_clahe,
    )


# [1차]: scan, gui, webcam, eda, evaluate, sweep 6개 서브커맨드를 정의하는 ArgumentParser를 구축합니다.
# [2차]: 스캐너가 받을 수 있는 6가지 명령어 메뉴판과 상세 옵션 안내서를 작성합니다.
def build_parser() -> argparse.ArgumentParser:
    """Construct top-level argument parser with subcommands."""
    # [1차]: 최상위 파서와 프로그램 설명문을 생성합니다.
    # [2차]: 안내 데스크 본체를 세웁니다.
    parser = argparse.ArgumentParser(
        prog="doc-scanner",
        description="OpenCV & NumPy 기반 규칙 기반 문서 스캐너 애플리케이션",
    )
    # [1차]: 서브커맨드 분기용 subparsers 컨테이너를 등록합니다.
    # [2차]: 6개 창구(scan, gui, webcam, eda, evaluate, sweep)를 개설합니다.
    sub = parser.add_subparsers(dest="command")

    # [1차]: 단일 이미지 헤드리스 스캔 서브커맨드(scan)와 옵션을 등록합니다.
    # [2차]: 1번 창구: 사진 경로를 넣으면 즉시 스캔해 저장하는 창구입니다.
    p_scan = sub.add_parser("scan", help="단일 이미지 문서 스캔 및 결과 저장")
    p_scan.add_argument("image", help="입력 이미지 경로")
    p_scan.add_argument("-o", "--output", default=None, help="최종 스캔 결과 저장 경로")
    p_scan.add_argument("--save-stages", action="store_true", help="6단계 몽타주 및 중간 이미지 저장")
    p_scan.add_argument("--stages-dir", default="outputs/stages", help="단계별 출력 폴더")
    p_scan.add_argument("--preset", choices=sorted(PRESETS.keys()), default="default")
    p_scan.add_argument("--blur", type=int, default=None, help="가우시안 블러 커널 크기(홀수)")
    p_scan.add_argument("--canny-low", type=int, default=None, help="Canny 하한 임계값")
    p_scan.add_argument("--canny-high", type=int, default=None, help="Canny 상한 임계값")
    p_scan.add_argument("--morph", type=int, default=None, help="모폴로지 닫힘 커널 크기")
    p_scan.add_argument("--clahe", action="store_true", help="CLAHE 국소 대비 보정 활성화")

    # [1차]: 실시간 트랙바 GUI 서브커맨드(gui)를 등록합니다.
    # [2차]: 2번 창구: 5단계 창과 실시간 트랙바를 띄우는 체험 창구입니다.
    p_gui = sub.add_parser("gui", help="실시간 트랙바 및 단계별 윈도우 GUI 실행")
    p_gui.add_argument("image", help="입력 이미지 경로")
    p_gui.add_argument("-o", "--output", default=None, help="'s' 키 저장 대상 경로")
    p_gui.add_argument("--preset", choices=sorted(PRESETS.keys()), default="default")

    # [1차]: 실시간 웹캠 스캔 서브커맨드(webcam)를 등록합니다.
    # [2차]: 3번 창구: 카메라로 실시간 문서를 스캔하는 창구입니다.
    p_cam = sub.add_parser("webcam", help="실시간 웹캠 문서 스캔 모드 (s: 저장, q: 종료)")
    p_cam.add_argument("--camera", type=int, default=0, help="웹캠 장치 인덱스 (기본 0)")
    p_cam.add_argument("--output-dir", default="outputs/webcam", help="캡처 저장 폴더")
    p_cam.add_argument("--preset", choices=sorted(PRESETS.keys()), default="default")

    # [1차]: 20장 밝기 히스토그램 분석 서브커맨드(eda)를 등록합니다.
    # [2차]: 4번 창구: 20장 사진의 밝기 분포를 진단하는 창구입니다.
    p_eda = sub.add_parser("eda", help="20장 테스트 데이터셋 밝기 히스토그램 EDA 수행")
    p_eda.add_argument(
        "--metadata",
        default="data/flow_document_images/metadata_flow.json",
        help="데이터셋 메타데이터 경로",
    )
    p_eda.add_argument("-o", "--output", default="outputs/eda_summary.json", help="EDA 요약 JSON 저장 경로")

    # [1차]: 20장 정량적 성능 평가 서브커맨드(evaluate)를 등록합니다.
    # [2차]: 5번 창구: 20장 시험지를 채점하고 조건별 성공률을 뽑는 창구입니다.
    p_eval = sub.add_parser("evaluate", help="20장 테스트셋 정량적 성능 평가 수행")
    p_eval.add_argument(
        "--metadata",
        default="data/flow_document_images/metadata_flow.json",
        help="데이터셋 메타데이터 경로",
    )
    p_eval.add_argument("-o", "--output", default="outputs/eval_summary.json", help="평가 JSON 저장 경로")
    p_eval.add_argument("--save-stages-dir", default=None, help="20장 전체 몽타주 저장 폴더")
    p_eval.add_argument("--preset", choices=sorted(PRESETS.keys()), default="default")

    # [1차]: 5개 프리셋 파라미터 딜레마 검증 서브커맨드(sweep)를 등록합니다.
    # [2차]: 6번 창구: 5가지 레시피를 모두 돌려 파라미터 딜레마를 입증하는 창구입니다.
    p_sweep = sub.add_parser("sweep", help="다중 프리셋 파라미터 스윕(파라미터 딜레마 검증)")
    p_sweep.add_argument(
        "--metadata",
        default="data/flow_document_images/metadata_flow.json",
        help="데이터셋 메타데이터 경로",
    )
    p_sweep.add_argument("-o", "--output", default="outputs/sweep_summary.json", help="스윕 JSON 저장 경로")

    # [1차]: 구성이 완료된 ArgumentParser 객체를 반환합니다.
    # [2차]: 완성된 안내 데스크 메뉴판을 돌려줍니다.
    return parser


# [1차]: 명령줄 인자를 해석해 해당 서브커맨드를 실행하고 POSIX 종료 코드(0/1/2)를 반환합니다.
# [2차]: 사용자가 입력한 명령에 맞춰 알맞은 창구로 안내하고 결과를 출력한 뒤 종료 코드를 돌려줍니다.
def main(argv: Optional[List[str]] = None) -> int:
    """Run CLI entry point and return POSIX exit code."""
    # [1차]: Windows 콘솔에서 한글 JSON/메시지가 깨지지 않도록 UTF-8로 설정합니다.
    # [2차]: 먼저 터미널 언어판을 UTF-8로 맞춥니다.
    configure_utf8_console()
    # [1차]: 테스트 주입 인자(argv) 또는 실제 sys.argv[1:]을 리스트로 복사합니다.
    # [2차]: 사용자가 터미널에 입력한 단어 목록을 받아옵니다.
    raw_args = list(sys.argv[1:] if argv is None else argv)

    # [1차]: 첫 인자가 알려진 서브커맨드가 아니면(즉시 이미지 경로 입력 시) 자동으로 'scan'을 앞에 삽입합니다.
    # [2차]: `python main.py photo.jpg`처럼 경로만 바로 입력해도 알아서 'scan' 명령으로 즉시 동작하게 배려합니다!
    known_commands = {"scan", "gui", "webcam", "eda", "evaluate", "sweep", "-h", "--help"}
    if raw_args and raw_args[0] not in known_commands:
        raw_args = ["scan", *raw_args]

    # [1차]: 파서를 생성하고 인자를 파싱합니다.
    # [2차]: 안내 데스크가 입력 단어를 깔끔하게 해석합니다.
    parser = build_parser()
    args = parser.parse_args(raw_args)

    # [1차]: 아무 인자 없이 실행된 경우 도움말을 출력하고 정상 종료(0)합니다.
    # [2차]: 빈손으로 오셨으면 사용법 안내문을 보여드리고 끝냅니다.
    if not args.command:
        parser.print_help()
        return 0

    # [1차]: 파일 누락, 손상 이미지, 카메라 미연결 등 예상 예외를 포착해 깔끔한 에러 메시지와 코드 1을 반환합니다.
    # [2차]: 도중에 문제가 생겨도 무서운 긴 에러 화면 대신 친절한 한 줄 안내문만 띄우고 안전 종료합니다.
    try:
        # [1차]: 'scan' 서브커맨드 — 단일 이미지 스캔, 결과 저장, 선택적 6단계 몽타주 저장 및 JSON 요약 출력.
        # [2차]: 1번 창구: 사진 1장을 스캔하고 결과 영수증을 출력합니다.
        if args.command == "scan":
            cfg = _build_config_from_args(args)
            scanner = DocumentScanner(cfg)
            result = scanner.scan_file(args.image, cfg)
            stem = Path(args.image).stem
            out_path = Path(args.output) if args.output else Path("outputs") / f"{stem}_scanned.png"

            saved_scan = None
            if result.detected and result.scanned is not None:
                saved_scan = str(save_image(out_path, result.scanned))

            stage_files = {}
            if args.save_stages:
                stage_files = {
                    k: str(v) for k, v in save_stage_outputs(result, args.stages_dir, stem).items()
                }

            payload = {
                "command": "scan",
                "input": str(args.image),
                "detected": result.detected,
                "area_ratio": round(result.area_ratio, 4),
                "failure_reason": result.failure_reason,
                "saved_scan": saved_scan,
                "stage_files": stage_files,
                "config": cfg.to_dict(),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            # [1차]: 검출 성공 시 0, 정상 실행했으나 문서 미검출 시 2를 반환합니다.
            # [2차]: 스캔 성공은 0, 문서를 못 찾은 경우는 구별하기 쉽게 2를 돌려줍니다.
            return 0 if result.detected else 2

        # [1차]: 'gui' 서브커맨드 — 5단계 창 + 실시간 트랙바 GUI를 실행합니다.
        # [2차]: 2번 창구: 실시간 트랙바 스튜디오를 엽니다.
        if args.command == "gui":
            cfg = _build_config_from_args(args)
            run_interactive_gui(args.image, initial_config=cfg, output_path=args.output)
            return 0

        # [1차]: 'webcam' 서브커맨드 — 웹캠 실시간 스캔 모드를 실행합니다.
        # [2차]: 3번 창구: 웹캠 카메라 스캔을 시작하고 저장한 파일 목록을 출력합니다.
        if args.command == "webcam":
            cfg = _build_config_from_args(args)
            saved = run_webcam_mode(camera_index=args.camera, config=cfg, output_dir=args.output_dir)
            print(json.dumps({"saved_files": [str(p) for p in saved]}, ensure_ascii=False, indent=2))
            return 0

        # [1차]: 'eda' 서브커맨드 — 20장 밝기 히스토그램 분석을 실행하고 요약 JSON을 출력합니다.
        # [2차]: 4번 창구: 20장 밝기 건강검진을 실행하고 핵심 수치를 화면에 띄웁니다.
        if args.command == "eda":
            summary = run_dataset_eda(args.metadata, output_json=args.output)
            compact = {
                "total_images": summary["total_images"],
                "difficulty_summary": {
                    k: {m: v for m, v in vals.items() if m != "mean_histogram"}
                    for k, vals in summary["difficulty_summary"].items()
                },
                "saved_json": args.output,
            }
            print(json.dumps(compact, ensure_ascii=False, indent=2))
            return 0

        # [1차]: 'evaluate' 서브커맨드 — 20장 정량적 성능 평가를 실행합니다.
        # [2차]: 5번 창구: 20장 시험을 채점해 조건별 합격률을 출력합니다.
        if args.command == "evaluate":
            cfg = _build_config_from_args(args)
            summary = evaluate_dataset(
                args.metadata,
                config=cfg,
                save_outputs_dir=args.save_stages_dir,
                output_json=args.output,
            )
            print(
                json.dumps(
                    {
                        "overall": summary["overall"],
                        "by_condition": summary["by_condition"],
                        "saved_json": args.output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        # [1차]: 'sweep' 서브커맨드 — 5개 프리셋 파라미터 스윕으로 파라미터 딜레마를 검증합니다.
        # [2차]: 6번 창구: 5가지 레시피 대결 결과를 출력합니다.
        if args.command == "sweep":
            sweep = run_parameter_sweep(args.metadata, output_json=args.output)
            compact = {
                preset: data["overall"] for preset, data in sweep["presets"].items()
            }
            print(
                json.dumps(
                    {
                        "preset_scores": compact,
                        "best_single_preset_success": sweep["best_single_preset_success"],
                        "oracle_union_success": sweep["oracle_union_success"],
                        "parameter_dilemma_proven": sweep["parameter_dilemma_proven"],
                        "saved_json": args.output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # [1차]: 예상된 운영 오류를 표준 에러(stderr)로 출력하고 종료 코드 1을 반환합니다.
        # [2차]: 빨간 에러 대신 깔끔한 [ERROR] 안내문을 띄우고 종료합니다.
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    # [1차]: 정상 종료 코드 0을 반환합니다.
    # [2차]: 문제없이 마쳤음을 알리며 끝냅니다.
    return 0
