"""Quantitative 20-image benchmark evaluator and parameter dilemma sweep runner."""

# [1차]: 타입 어노테이션 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 나중에 해석하도록 미리 켜둡니다.
from __future__ import annotations

# [1차]: 평가 레코드 정의(@dataclass)와 딕셔너리 변환(asdict) 도구를 가져옵니다.
# [2차]: 1문제 채점표 틀과 JSON 변환기를 꺼냅니다.
from dataclasses import asdict, dataclass

# [1차]: 파일 경로 조합을 위해 Path를 가져옵니다.
# [2차]: 사진 파일 주소를 깔끔하게 다루는 주소 객체입니다.
from pathlib import Path

# [1차]: 딕셔너리·리스트·선택적 타입 힌트를 가져옵니다.
# [2차]: 입출력 데이터 모양을 알려주는 안내 라벨들입니다.
from typing import Any, Dict, List, Optional

# [1차]: 정답 좌표 배열 변환을 위해 NumPy를 가져옵니다.
# [2차]: 정답 4꼭짓점 숫자를 계산용 행렬로 바꿉니다.
import numpy as np

# [1차]: 기본 설정과 5개 비교 프리셋(PRESETS)을 config 모듈에서 가져옵니다.
# [2차]: 5가지 추천 레시피 세트를 불러옵니다.
from .config import PRESETS, ScanConfig

# [1차]: 정량 채점 지표인 꼭짓점 오차와 사각형 IoU 계산 함수를 geometry 모듈에서 가져옵니다.
# [2차]: 정답 네모와 얼마나 겹치는지(IoU), 몇 픽셀 빗나갔는지 재는 채점 자를 꺼냅니다.
from .geometry import compute_corner_error, compute_quad_iou

# [1차]: 한글 안전 이미지·JSON 입출력 유틸리티를 가져옵니다.
# [2차]: 다국어 택배 기사를 불러옵니다.
from .io_utils import PathLike, load_image, load_json, save_json

# [1차]: 스캐너 오케스트레이터와 단계별 시각화 저장 함수를 pipeline 모듈에서 가져옵니다.
# [2차]: 시험 문제를 풀 스캐너 본체와 6컷 포스터 저장기를 불러옵니다.
from .pipeline import DocumentScanner, save_stage_outputs

# [1차]: 과제 보고서 표 규격과 100% 일치하는 4개 조건의 고정 출력 순서입니다.
# [2차]: 성적표 4줄이 항상 과제 안내서와 똑같은 순서로 나오도록 순서를 못 박아 둡니다.
CONDITION_ORDER = [
    "단순 배경 + 정면",
    "그림자 있음",
    "45도 이상 기울어짐",
    "복잡한 배경",
]


# [1차]: 테스트 이미지 1장에 대한 검출 여부, 정량 통과 여부, IoU, 꼭짓점 오차, 실패 사유 레코드입니다.
# [2차]: 20문제 중 1문제의 O/X 채점 결과와 상세 오답 노트를 담는 카드입니다.
@dataclass
class SingleEvalRecord:
    """Quantitative evaluation metrics for a single test image."""

    # [1차]: 이미지 고유 ID입니다.
    # [2차]: 문제 번호(예: easy_01, shadow_04)입니다.
    image_id: str
    # [1차]: 원본 이미지 파일명입니다.
    # [2차]: 시험지 파일 이름입니다.
    filename: str
    # [1차]: 소속 촬영 조건입니다.
    # [2차]: 4개 시험 유형 중 어디에 속하는지 표시합니다.
    condition: str
    # [1차]: 파이프라인이 4각형 후보를 검출했는지 여부입니다.
    # [2차]: 일단 네모 답안을 적어냈는지(True) 백지로 냈는지(False) 표시합니다.
    detected: bool
    # [1차]: 정답 대비 IoU·꼭짓점 오차 기준까지 모두 통과한 최종 성공 여부입니다.
    # [2차]: 진짜 정답 종이를 정확히 맞춰서 최종 합격(O)인지 불합격(X)인지 도장입니다.
    success: bool
    # [1차]: 예측 사각형과 정답 사각형의 IoU 점수(0.0~1.0)입니다.
    # [2차]: 정답 영역과 몇 % 겹쳤는지 나타내는 정확도 점수입니다.
    iou: float
    # [1차]: 정렬된 4꼭짓점 평균 거리 오차(px, 미검출 시 None)입니다.
    # [2차]: 모서리 핀이 정답에서 평균 몇 픽셀 어긋났는지 재는 오차입니다.
    corner_error_px: Optional[float]
    # [1차]: 검출된(또는 최대 탈락) 윤곽선의 화면 면적 비율입니다.
    # [2차]: 찾은 영역이 화면의 몇 % 크기였는지 기록합니다.
    area_ratio: float
    # [1차]: 실패 시 원인 진단 문자열(성공 시 None)입니다.
    # [2차]: 왜 틀렸는지 적어둔 오답 노트 사유입니다.
    failure_reason: Optional[str]


# [1차]: 지정된 ScanConfig로 20장 데이터셋 전체를 채점하고 조건별·전체 성공률을 산출합니다.
# [2차]: 스캐너에게 20문제 시험지를 풀게 한 뒤, 4개 유형별 성적표와 오답 노트를 자동으로 뽑아줍니다.
def evaluate_dataset(
    metadata_path: PathLike,
    config: Optional[ScanConfig] = None,
    iou_threshold: float = 0.75,
    max_corner_error_px: float = 45.0,
    save_outputs_dir: Optional[PathLike] = None,
    output_json: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Evaluate scanner accuracy across all labeled dataset images."""
    # [1차]: 메타데이터 JSON을 로드하고 설정과 스캐너 인스턴스를 준비합니다.
    # [2차]: 시험지 목록과 정답지(metadata.json)를 펼치고 스캐너를 준비합니다.
    meta_file = Path(metadata_path)
    meta = load_json(meta_file)
    base_dir = meta_file.parent
    cfg = config or ScanConfig()
    scanner = DocumentScanner(cfg)

    # [1차]: 20개 개별 채점 레코드를 누적할 리스트입니다.
    # [2차]: 한 문제씩 채점한 결과를 모아둘 성적표 바구니입니다.
    records: List[SingleEvalRecord] = []

    # [1차]: 모든 테스트 이미지를 순회하며 스캔 실행 및 정답 대비 정량 채점을 수행합니다.
    # [2차]: 1번부터 20번까지 사진을 차례로 스캔하고 정답 모서리와 비교해 채점합니다.
    for item in meta["images"]:
        img_path = base_dir / item["relative_path"]
        image = load_image(img_path)
        result = scanner.scan_image(image, cfg)

        # [1차]: 메타데이터의 정답 4꼭짓점 좌표를 float32 배열로 변환하고 지표를 초기화합니다.
        # [2차]: 정답지에 적힌 진짜 네 모서리 좌표를 꺼내 채점을 준비합니다.
        gt_corners = np.asarray(item["gt_corners"], dtype=np.float32)
        iou = 0.0
        corner_err: Optional[float] = None
        success = False
        reason = result.failure_reason

        # [1차]: 4각형이 검출되었다면 IoU와 꼭짓점 오차를 계산해 엄격한 성공 기준을 판정합니다.
        # [2차]: 아무 네모나 찾았다고 점수를 주지 않고, 정답 종이와 75% 이상 겹치고 45px 이내일 때만 합격시킵니다!
        if result.detected and result.ordered_quad is not None:
            iou = compute_quad_iou(result.ordered_quad, gt_corners, image.shape[:2])
            corner_err = compute_corner_error(result.ordered_quad, gt_corners)
            if iou >= iou_threshold and corner_err <= max_corner_error_px:
                success = True
                reason = None
            else:
                success = False
                reason = f"inaccurate_quad(iou={iou:.3f}, err={corner_err:.1f}px)"

        # [1차]: 시각화 저장 폴더가 지정된 경우 각 이미지의 6패널 몽타주와 중간 이미지를 저장합니다.
        # [2차]: 저장 옵션이 켜져 있으면 모든 문제의 풀이 과정 6컷 포스터를 앨범에 보관합니다.
        if save_outputs_dir is not None:
            save_stage_outputs(result, save_outputs_dir, stem=item["id"])

        # [1차]: 개별 평가 레코드를 반올림 정리해 목록에 추가합니다.
        # [2차]: 방금 채점한 문제의 성적 카드를 바구니에 넣습니다.
        records.append(
            SingleEvalRecord(
                image_id=item["id"],
                filename=item["filename"],
                condition=item["condition"],
                detected=result.detected,
                success=success,
                iou=round(iou, 4),
                corner_error_px=round(corner_err, 2) if corner_err is not None else None,
                area_ratio=round(result.area_ratio, 4),
                failure_reason=reason,
            )
        )

    # [1차]: 과제 표의 고정 조건 순서대로 조건별 성공/실패/성공률 행을 집계합니다.
    # [2차]: 4개 시험 유형별로 "몇 문제 중 몇 개 맞았는지" 표의 각 줄을 계산합니다.
    condition_rows: List[Dict[str, Any]] = []
    seen_conditions = [c for c in CONDITION_ORDER if any(r.condition == c for r in records)]
    for cond in sorted({r.condition for r in records}):
        if cond not in seen_conditions:
            seen_conditions.append(cond)

    for cond in seen_conditions:
        subset = [r for r in records if r.condition == cond]
        passed = sum(1 for r in subset if r.success)
        failed = len(subset) - passed
        rate = round((passed / len(subset)) * 100.0, 1) if subset else 0.0
        condition_rows.append(
            {
                "condition": cond,
                "total": len(subset),
                "success": passed,
                "fail": failed,
                "success_rate": rate,
            }
        )

    # [1차]: 전체 20장 합계 통계를 구성하고 JSON 저장 옵션 처리 후 반환합니다.
    # [2차]: 최종 총점과 전체 합격률을 계산해 성적표 파일로 저장하고 돌려줍니다.
    total_count = len(records)
    total_pass = sum(1 for r in records if r.success)
    summary: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "criteria": {
            "iou_threshold": iou_threshold,
            "max_corner_error_px": max_corner_error_px,
        },
        "overall": {
            "total": total_count,
            "success": total_pass,
            "fail": total_count - total_pass,
            "success_rate": round((total_pass / total_count) * 100.0, 1) if total_count else 0.0,
        },
        "by_condition": condition_rows,
        "records": [asdict(r) for r in records],
    }
    if output_json is not None:
        save_json(output_json, summary)
    return summary


# [1차]: 여러 프리셋을 20장 전체에 교차 실행해 상충 사례(Conflict Pairs)와 Oracle 상한을 계산합니다.
# [2차]: 5가지 레시피로 똑같은 20문제 시험을 5번 치러 "A를 살리면 B가 죽는 파라미터 딜레마"를 숫자로 증명합니다.
def run_parameter_sweep(
    metadata_path: PathLike,
    presets: Optional[Dict[str, ScanConfig]] = None,
    output_json: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Run multiple parameter presets over the 20 images to expose the Parameter Dilemma."""
    # [1차]: 지정 프리셋이 없으면 config.PRESETS 5종 전체를 실험 대상으로 사용합니다.
    # [2차]: 준비된 5가지 대표 레시피를 모두 시험장에 올립니다.
    preset_map = presets or PRESETS
    preset_results: Dict[str, Dict[str, Any]] = {}
    per_image_matrix: Dict[str, Dict[str, bool]] = {}

    # [1차]: 각 프리셋마다 evaluate_dataset을 수행하고 이미지×프리셋 2차원 성공 행렬을 채웁니다.
    # [2차]: 5가지 레시피로 20장씩 채점해 "어느 레시피가 어느 사진을 맞혔는지" 바둑판 표를 만듭니다.
    for preset_name, cfg in preset_map.items():
        eval_res = evaluate_dataset(metadata_path, config=cfg)
        preset_results[preset_name] = {
            "config": cfg.to_dict(),
            "overall": eval_res["overall"],
            "by_condition": eval_res["by_condition"],
        }
        for rec in eval_res["records"]:
            per_image_matrix.setdefault(rec["image_id"], {})[preset_name] = bool(rec["success"])

    # [1차]: 모든 프리셋 쌍(A, B)에 대해 A만 성공하고 B는 실패한 이미지와 그 반대 사례가 동시에 존재하는지 탐색합니다.
    # [2차]: "A 레시피는 성공하는데 B는 실패하고, 반대로 다른 사진에선 B만 성공하는" 모순 쌍을 자동으로 찾아냅니다.
    conflict_pairs: List[Dict[str, Any]] = []
    preset_names = list(preset_map.keys())
    for i in range(len(preset_names)):
        for j in range(i + 1, len(preset_names)):
            p_a, p_b = preset_names[i], preset_names[j]
            only_a = [img for img, res in per_image_matrix.items() if res[p_a] and not res[p_b]]
            only_b = [img for img, res in per_image_matrix.items() if res[p_b] and not res[p_a]]
            if only_a and only_b:
                conflict_pairs.append(
                    {
                        "preset_a": p_a,
                        "preset_b": p_b,
                        "only_a_succeeds": only_a,
                        "only_b_succeeds": only_b,
                    }
                )

    # [1차]: 이미지마다 최적 프리셋을 사후 선택한 Oracle 합집합 점수와 단일 최고 프리셋 점수를 비교합니다.
    # [2차]: "하나의 고정 설정 최고 점수"와 "사진마다 신처럼 맞춤 설정을 고른 점수"의 격차를 계산합니다.
    oracle_success = sum(1 for res in per_image_matrix.values() if any(res.values()))
    total_images = len(per_image_matrix)
    best_single = max((v["overall"]["success"] for v in preset_results.values()), default=0)

    # [1차]: 스윕 결과를 종합해 parameter_dilemma_proven 불리언 지표와 함께 반환·저장합니다.
    # [2차]: 파라미터 딜레마가 수학적으로 입증되었는지 최종 판정표로 정리해 돌려줍니다.
    sweep_summary: Dict[str, Any] = {
        "presets": preset_results,
        "per_image_matrix": per_image_matrix,
        "best_single_preset_success": best_single,
        "oracle_union_success": oracle_success,
        "total_images": total_images,
        "parameter_dilemma_proven": oracle_success > best_single and len(conflict_pairs) > 0,
        "conflict_pairs": conflict_pairs,
    }
    if output_json is not None:
        save_json(output_json, sweep_summary)
    return sweep_summary
