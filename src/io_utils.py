"""Unicode-safe image and JSON I/O utilities for cross-platform Windows/Linux support."""

# [1차]: 타입 힌트 문자열 지연 평가를 활성화합니다.
# [2차]: 타입 이름표를 나중에 해석하도록 미리 켜두는 스위치입니다.
from __future__ import annotations

# [1차]: 메타데이터 및 분석 결과 JSON 직렬화를 위해 표준 json 모듈을 가져옵니다.
# [2차]: 실험 결과를 깔끔한 장부 파일(.json)로 읽고 쓰는 도구를 챙깁니다.
import json

# [1차]: 표준 출력/에러 스트림의 인코딩 재설정을 위해 sys 모듈을 가져옵니다.
# [2차]: 터미널 화면 글자 깨짐을 고치기 위해 콘솔 창구(stdout/stderr)를 불러옵니다.
import sys

# [1차]: OS 독립적 파일 경로 조작을 위해 pathlib.Path를 가져옵니다.
# [2차]: 윈도우(`\`)와 맥/리눅스(`/`) 경로 차이를 알아서 맞춰주는 스마트 주소 객체입니다.
from pathlib import Path

# [1차]: 범용 타입 힌트 Any와 Union을 가져옵니다.
# [2차]: "문자열이든 Path 객체든 다 받을 수 있어"라고 알려주는 안내 라벨입니다.
from typing import Any, Union

# [1차]: 이미지 인코딩·디코딩을 위해 OpenCV(cv2)를 가져옵니다.
# [2차]: 이미지 파일을 숫자 배열로 풀고 다시 사진 파일로 압축하는 엔진입니다.
import cv2

# [1차]: 파일 바이트를 uint8 1차원 배열로 직접 읽기 위해 NumPy를 가져옵니다.
# [2차]: 한글 경로를 우회해 파일 원시 데이터를 그대로 담아올 바구니(NumPy)입니다.
import numpy as np

# [1차]: 문자열 경로와 Path 객체를 모두 허용하는 경로 타입 별칭을 정의합니다.
# [2차]: 글자로 쓴 주소든 경로 객체든 똑같이 받겠다는 약속입니다.
PathLike = Union[str, Path]


# [1차]: Windows 콘솔(CP949 등)에서 한글 출력 시 UnicodeEncodeError를 방지하기 위해 UTF-8로 재설정합니다.
# [2차]: 터미널이 한글을 못 알아듣고 에러를 내지 않도록 출력 언어판을 UTF-8로 바꿔 끼웁니다.
def configure_utf8_console() -> None:
    """Ensure stdout and stderr emit UTF-8 cleanly on Windows consoles."""
    # [1차]: 표준 출력(stdout)과 표준 에러(stderr) 두 스트림을 순회합니다.
    # [2차]: 일반 안내 방송과 경고 방송 두 스피커를 차례대로 점검합니다.
    for stream in (sys.stdout, sys.stderr):
        # [1차]: 스트림 객체에 reconfigure 메서드가 있는지 안전하게 조회합니다.
        # [2차]: 스피커에 언어 변경 버튼(reconfigure)이 달려 있는지 확인합니다.
        reconfigure = getattr(stream, "reconfigure", None)
        # [1차]: 호출 가능한 메서드라면 UTF-8 인코딩 적용을 시도합니다.
        # [2차]: 버튼이 실제로 작동하면 UTF-8 모드로 눌러줍니다.
        if callable(reconfigure):
            try:
                # [1차]: 인코딩을 utf-8로 설정하고, 변환 불가 문자는 에러 대신 대체 문자로 처리합니다.
                # [2차]: 한글을 완벽히 출력하되, 혹시 모를 이상한 기호가 와도 멈추지 않게 안전망을 칩니다.
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                # [1차]: 리다이렉트된 파이프 등 재설정이 불가능한 환경에서는 조용히 무시합니다.
                # [2차]: 버튼이 잠긴 특수 환경에서는 굳이 무리하지 않고 넘어갑니다.
                pass


# [1차]: 한글·공백 포함 경로에서도 안전하게 BGR 컬러 이미지를 로드하는 함수입니다.
# [2차]: OpenCV 기본 imread가 한글 폴더에서 실패하는 함정을 피해, 파이썬이 파일을 먼저 읽어 OpenCV에 건네줍니다.
def load_image(path: PathLike) -> np.ndarray:
    """Load an image from any Unicode/Korean file path safely via NumPy buffer decoding."""
    # [1차]: 입력 경로를 Path 객체로 통일합니다.
    # [2차]: 문자열 주소를 다루기 편한 스마트 주소 객체로 바꿉니다.
    file_path = Path(path)
    # [1차]: 파일이 존재하지 않거나 디렉터리인 경우 명확한 FileNotFoundError를 발생시킵니다.
    # [2차]: 사진 파일이 실제로 그 자리에 있는지 먼저 확인하고, 없으면 친절히 알려줍니다.
    if not file_path.exists() or not file_path.is_file():
        # [1차]: 누락된 경로를 포함한 예외를 던져 호출자가 즉시 원인을 알 수 있게 합니다.
        # [2차]: "요청한 사진이 서랍에 없어요!"라고 명확한 안내문을 띄웁니다.
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {file_path}")

    # [1차]: np.fromfile로 파일의 원시 바이트 전체를 uint8 1차원 NumPy 배열로 읽습니다.
    # [2차]: 한글 폴더 주소는 파이썬/NumPy가 읽어서 파일 알맹이(바이트)만 상자에 담습니다.
    raw_bytes = np.fromfile(str(file_path), dtype=np.uint8)
    # [1차]: 0바이트 빈 파일인지 검사하여 빈 배열 디코딩 오류를 사전에 차단합니다.
    # [2차]: 택배 상자를 열어봤는데 속이 텅 비어 있으면 바로 경고합니다.
    if raw_bytes.size == 0:
        # [1차]: 빈 파일임을 알리는 ValueError를 발생시킵니다.
        # [2차]: "빈 껍데기 파일이라 읽을 수 없어요!"라고 알려줍니다.
        raise ValueError(f"이미지 파일이 비어 있습니다: {file_path}")

    # [1차]: 바이트 버퍼를 OpenCV imdecode로 해독하여 (H, W, 3) BGR 배열로 복원합니다.
    # [2차]: 압축된 JPG/PNG 바이트 상자를 OpenCV가 풀어 픽셀 도화지(3차원 배열)로 펼칩니다.
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    # [1차]: 이미지가 손상되었거나 지원하지 않는 형식이라 None이 반환된 경우 예외를 발생시킵니다.
    # [2차]: 그림 파일이 깨졌거나 텍스트 파일에 .jpg 이름표만 붙인 가짜 파일이면 즉시 거절합니다.
    if image is None or image.size == 0:
        # [1차]: 디코딩 실패 사유와 경로를 담아 ValueError를 발생시킵니다.
        # [2차]: "손상된 이미지라 열 수 없어요!"라고 구체적으로 알립니다.
        raise ValueError(f"이미지를 디코딩할 수 없습니다 (손상되었거나 미지원 형식): {file_path}")

    # [1차]: 정상 디코딩된 (Height, Width, 3) uint8 NumPy 배열을 반환합니다.
    # [2차]: 준비가 끝난 픽셀 도화지를 다음 공정으로 넘깁니다.
    return image


# [1차]: 한글·공백 포함 경로에서도 안전하게 이미지 배열을 디스크에 저장하는 함수입니다.
# [2차]: 완성된 스캔 이미지를 한글 폴더에도 깨짐 없이 안전하게 포장해서 배송(저장)합니다.
def save_image(path: PathLike, image: np.ndarray) -> Path:
    """Save an image to any Unicode/Korean file path safely via cv2.imencode + tofile."""
    # [1차]: 저장할 객체가 None이거나 유효한 비어 있지 않은 NumPy 배열인지 검증합니다.
    # [2차]: 빈 도화지를 저장하려 하면 미리 막아줍니다.
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        # [1차]: 빈 배열 저장 시도를 ValueError로 차단합니다.
        # [2차]: "저장할 그림 내용이 비어 있습니다!"라고 안내합니다.
        raise ValueError("저장할 이미지 배열이 비어 있습니다.")

    # [1차]: 대상 경로를 Path 객체로 변환합니다.
    # [2차]: 저장할 주소를 스마트 주소 객체로 정리합니다.
    out_path = Path(path)
    # [1차]: 상위 디렉터리가 없으면 중첩 폴더까지 모두 자동 생성합니다.
    # [2차]: 택배를 넣을 서랍장이 아직 없으면 먼저 서랍장부터 만들어 둡니다.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # [1차]: 확장자를 소문자로 추출하고, 확장자가 없으면 기본값 .png를 사용합니다.
    # [2차]: 파일 이름 끝에 어떤 사진 형식(.png/.jpg)으로 포장할지 확인합니다.
    ext = out_path.suffix.lower() if out_path.suffix else ".png"
    # [1차]: 지원하지 않는 확장자가 들어온 경우 안전하게 .png로 교정합니다.
    # [2차]: 알 수 없는 포장지 이름이 오면 가장 안전한 무손실 .png 포장지로 바꿉니다.
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        # [1차]: 확장자를 .png로 고정합니다.
        # [2차]: 안전 확장자 .png를 지정합니다.
        ext = ".png"
        # [1차]: 출력 경로 객체의 확장자도 함께 갱신합니다.
        # [2차]: 파일 주소 끝의 이름표도 .png로 고쳐 적습니다.
        out_path = out_path.with_suffix(ext)

    # [1차]: 메모리 상에서 이미지 배열을 지정한 포맷의 바이트 버퍼로 인코딩합니다.
    # [2차]: 픽셀 도화지를 메모리 안에서 먼저 JPG/PNG 압축 파일로 포장합니다.
    success, encoded = cv2.imencode(ext, image)
    # [1차]: 인코딩 실패 시 ValueError를 발생시킵니다.
    # [2차]: 포장에 실패했다면 그대로 넘어가며 빈 파일을 남기지 않고 에러를 알립니다.
    if not success:
        # [1차]: 실패한 확장자와 경로 정보를 포함해 예외를 던집니다.
        # [2차]: "이 형식으로는 사진 포장에 실패했어요!"라고 보고합니다.
        raise ValueError(f"이미지 인코딩에 실패했습니다 ({ext}): {out_path}")

    # [1차]: 인코딩된 바이트 배열을 NumPy tofile()로 유니코드 경로에 직접 기록합니다.
    # [2차]: 한글 경로도 문제없이 다루는 NumPy 택배 기사가 완성된 상자를 디스크에 내려놓습니다.
    encoded.tofile(str(out_path))
    # [1차]: 실제로 저장된 최종 Path 객체를 반환합니다.
    # [2차]: "여기 주소에 잘 보관했습니다"라고 영수증(경로)을 건네줍니다.
    return out_path


# [1차]: 파이썬 자료구조를 한글이 깨지지 않는 UTF-8 JSON 파일로 저장합니다.
# [2차]: 실험 결과 장부를 사람이 읽기 좋게 들여쓰기해서 한글 그대로 파일에 적습니다.
def save_json(path: PathLike, data: Any) -> Path:
    """Write JSON data with UTF-8 encoding and readable indentation."""
    # [1차]: 저장 경로를 Path 객체로 변환합니다.
    # [2차]: 주소를 표준 경로 객체로 준비합니다.
    out_path = Path(path)
    # [1차]: 상위 폴더가 없으면 자동으로 생성합니다.
    # [2차]: 서류를 넣을 폴더가 없으면 새로 만듭니다.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # [1차]: ensure_ascii=False로 한글을 유니코드 이스케이프 없이 그대로 들여쓰기 저장합니다.
    # [2차]: 한글이 `\uac00`처럼 암호문으로 깨지지 않고 예쁜 한글 그대로 저장되게 합니다.
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    # [1차]: 저장 완료된 파일 경로를 반환합니다.
    # [2차]: 저장 위치 영수증을 돌려줍니다.
    return out_path


# [1차]: 디스크의 UTF-8 JSON 파일을 읽어 파이썬 객체(dict/list)로 복원합니다.
# [2차]: 저장해 둔 실험 장부(.json)를 펼쳐 파이썬이 바로 계산할 수 있는 데이터로 가져옵니다.
def load_json(path: PathLike) -> Any:
    """Read UTF-8 JSON data from disk."""
    # [1차]: 입력 경로를 Path 객체로 변환합니다.
    # [2차]: 주소를 Path 객체로 바꿉니다.
    in_path = Path(path)
    # [1차]: 파일이 존재하지 않으면 FileNotFoundError를 발생시킵니다.
    # [2차]: 장부 파일이 제자리에 없으면 바로 알려줍니다.
    if not in_path.exists():
        # [1차]: 누락된 JSON 경로를 포함한 예외를 발생시킵니다.
        # [2차]: "장부 파일이 보이지 않아요!"라고 보고합니다.
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {in_path}")
    # [1차]: UTF-8로 텍스트를 읽고 json.loads로 파싱한 결과를 반환합니다.
    # [2차]: 글자를 읽어 파이썬 딕셔너리·리스트 형태로 해독해 돌려줍니다.
    return json.loads(in_path.read_text(encoding="utf-8"))
