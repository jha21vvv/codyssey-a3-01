# 📋 PLAN.md — OpenCV 기반 지능형 문서 스캐너 설계 및 실행 계획서

> **프로젝트명**: `codyssey-a3-01` — OpenCV & NumPy 기반 규칙 기반 문서 스캐너 및 정량적 한계 분석 시스템  
> **작성 기준**: `코디세이 a3-1.md` 요구사항 100% 반영 + `PROJECT_WORKFLOW_PROMPT_TEMPLATE.md` 8단계 표준 프로세스

---

## 1. 문제 정의 및 핵심 목표

스마트폰 등으로 비스듬하게 촬영된 문서(A4 용지, 영수증, 명함 등)에서 **딥러닝/외부 OCR/외부 기하 라이브러리 없이** 순수 OpenCV와 NumPy의 규칙 기반 이미지 처리 파이프라인을 설계하여:

1. **원본 → 전처리(Grayscale + Gaussian Blur) → 엣지 검출(Canny) → 윤곽선 검출(`findContours` + `approxPolyDP`) → 수동 좌표 정렬 및 투시 변환(`getPerspectiveTransform` + `warpPerspective`) → 적응형 이진화 후처리(`adaptiveThreshold`)** 전 과정을 자동화한다.
2. **단계별 윈도우 시각화 + 실시간 트랙바(Blur 커널 크기, Canny low/high)** 및 **웹캠 실시간 스캔 모드(`s` 저장, `q` 종료)**를 제공하며, GUI가 불가능한 환경(Colab/CI/Headless)에서도 CLI 인자 및 파일 저장으로 100% 동작하게 설계한다.
3. **20장 벤치마크 테스트셋(4개 조건 × 5장)**에 대한 **이미지 탐색적 분석(EDA: 밝기 히스토그램 비교)** 및 **정량적 성능 평가(성공률, 파라미터 스윕, 실패 원인 분석)**를 수행하여 **"규칙 기반 접근의 한계(파라미터 딜레마)"**와 **"학습 기반 접근(딥러닝)이 필요한 이유"**를 데이터로 증명한다.

---

## 2. 제약 조건 및 준수 규격

| 구분 | 항목 | 준수 정책 |
| :--- | :--- | :--- |
| **개발 환경** | Python 3.8+ (현재 3.11.5) | 표준 라이브러리 + 허용 패키지만 사용 |
| **허용 필수** | `opencv-python>=4.5`, `numpy>=1.20` | 이미지 처리, 배열 연산, 기하 변환 |
| **허용 선택** | `matplotlib>=3.3` | EDA 히스토그램 및 아키텍처 다이어그램 PNG 생성 |
| **금지 사항** | PyTorch, TensorFlow, Keras | **일체 미사용** (자동 검증 스크립트로 import 검사) |
| **금지 사항** | Tesseract OCR 등 OCR 라이브러리 | **일체 미사용** |
| **금지 사항** | `scipy.spatial`, `shapely` 등 기하 라이브러리 | **일체 미사용** — 좌표 정렬은 순수 NumPy/Python 기하 원리로 직접 구현 |
| **경로 안전성** | Windows 한글/공백 경로 (`안재현` 등) | `cv2.imread`/`imwrite` 직접 호출 금지 → `np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `.tofile` 래퍼 강제 |
| **이미지 임베딩** | 마크다운 문서 내 PNG 링크 | 절대경로(`file:///...`) 금지 → 저장소 루트 상대경로(`diagrams/*.png`)만 사용 |

---

## 3. 단일 책임 원칙(SRP) 기반 모듈 아키텍처

```text
codyssey-a3-01/
├── main.py                              # 최상위 CLI 진입점
├── src/
│   ├── __init__.py                      # 패키지 공개 API export
│   ├── config.py                        # ScanConfig 데이터클래스, 파라미터 정규화/검증 (홀수 커널, 임계값 범위)
│   ├── io_utils.py                      # 한글/유니코드 안전 이미지 로드·저장, UTF-8 콘솔 설정, JSON 직렬화
│   ├── preprocessor.py                  # Grayscale 변환, 가우시안 블러, Canny 엣지 검출, 형태학적 닫힘(선택)
│   ├── contour_detector.py              # 닫힌 윤곽선 추출, 면적/종횡비 필터링, approxPolyDP 사각형 탐색
│   ├── geometry.py                      # 중심점 편각 기반 수동 4점 정렬(TL-TR-BR-BL) + 투시 변환 행렬 및 워핑
│   ├── postprocessor.py                 # 적응형 이진화(Adaptive Thresholding), 스캔 품질 보정, 결과 합성
│   ├── pipeline.py                      # 전체 파이프라인 오케스트레이터, 5단계 시각화 몽타주 생성
│   ├── gui.py                           # 단계별 OpenCV 윈도우, 실시간 트랙바, 웹캠 실시간 스캔('s'/'q')
│   ├── eda.py                           # 20장 밝기 히스토그램/통계 분석, 쉬운 vs 어려운 케이스 분포 비교
│   ├── evaluator.py                     # 20장 정량 평가(꼭짓점 오차/IoU/성공률) + 파라미터 스윕(딜레마 실험)
│   └── cli.py                           # CLI 서브커맨드 (scan, gui, webcam, eda, evaluate, sweep)
├── scripts/
│   ├── generate_test_dataset.py         # 재현 가능한 20장 현실적 테스트 데이터셋(4조건×5장) + metadata.json 생성
│   ├── generate_diagrams_matplotlib.py  # 고해상도 Matplotlib PNG 다이어그램 4종 생성
│   ├── verify_requirements.py           # 평가자 원클릭 시각적 검증 러너 (테이블 + JSON 출력)
│   └── build_notebooklm_source.py       # src/*.py 순수 소스코드 → NOTEBOOKLM_FULL_SOURCE.txt 자동 빌드
├── tests/
│   └── test_scanner.py                  # 초고속 Mock 기반 단위 테스트 (GUI/웹캠/엣지케이스 포함, 100% 통과)
├── data/
│   ├── test_images/                     # 20장 테스트 이미지 (easy/shadow/tilt/complex 각 5장)
│   └── metadata.json                    # 각 이미지의 촬영 조건 및 정답 4꼭짓점 좌표
├── diagrams/                            # Matplotlib 생성 PNG 다이어그램 (상대경로 임베딩)
├── outputs/                             # 스캔 결과, 단계별 시각화, EDA/평가 JSON·요약
└── study/
    └── 핵심_코드_스터디.md                # 핵심 코드 8선 단어분해 + 비유 2단 주석 학습서
```

---

## 4. 핵심 알고리즘 및 예외 처리 규격

### 4.1 수동 4점 좌표 정렬 (`src/geometry.py: order_points`)
- **기존 단순 `sum`/`diff` 방식의 한계**: 45도 회전된 마름모 형태 문서에서는 `x+y` 또는 `y-x` 값이 동률(Tie)이 되어 중복 꼭짓점을 선택하거나 나비넥타이(self-intersecting quad) 왜곡이 발생한다.
- **설계한 순수 기하 알고리즘 (Zero 외부 기하 라이브러리)**:
  1. 4개 점의 무게중심 `(cx, cy) = (mean(x), mean(y))` 계산.
  2. 화면 좌표계(y축 아래 방향)에서 각 점의 편각 `atan2(y - cy, x - cx)`을 계산하여 시계 방향 순환 순서 확정 (동률 시 `(x + y, y, x)` 사전순 타이 브레이킹).
  3. 순환 목록 중 좌상단(Top-Left) 기준점(최소 `(x + y, y, x)`)을 찾아 인덱스 `0`으로 회전(Rotate)시켜 `[TL, TR, BR, BL]` 4개 고유 좌표를 100% 보장.

### 4.2 예외 처리 규격
| 예외 상황 | 발생 모듈 | 방어 전략 |
| :--- | :--- | :--- |
| 존재하지 않는 경로 / 손상된 이미지 | `io_utils.py` | `FileNotFoundError` / `ValueError` 명확한 한글·영문 메시지 발생 |
| Windows 한글/공백 파일 경로 | `io_utils.py` | `np.fromfile` + `cv2.imdecode`, `cv2.imencode` + `.tofile` 사용 |
| 짝수 또는 음수 블러 커널 크기 | `config.py` | `>= 1` 홀수로 자동 정규화 (`k if k % 2 == 1 else k + 1`) |
| `canny_low > canny_high` 역전 | `config.py` | 자동 스왑(swap) 또는 최소 간격 보장 |
| 문서 윤곽선(4꼭짓점) 미검출 | `contour_detector.py` / `pipeline.py` | 크래시 없이 `detected=False` 반환, 원본 유지 및 실패 사유 기록 |
| 4점이 일직선(면적 0)이거나 중복점 | `geometry.py` | `ValueError` 검증 및 워핑 출력 크기 최소 `1x1` 클램핑 |
| GUI 미지원 환경 / 웹캠 미연결 | `gui.py` | `RuntimeError` 친절 안내 + 헤드리스 저장 모드 폴백 지원 |

---

## 5. CLI 인터페이스 명세

```bash
# 1. 단일 이미지 즉시 스캔 (헤드리스 저장 + 단계별 몽타주 출력)
python main.py scan data/test_images/easy_01.jpg --output outputs/scanned_easy_01.png --save-stages

# 2. 실시간 트랙바 + 단계별 윈도우 GUI 실행
python main.py gui data/test_images/easy_01.jpg

# 3. 웹캠 실시간 문서 스캔 ('s': 캡처 저장, 'q': 종료)
python main.py webcam --camera 0 --output-dir outputs/webcam

# 4. 20장 테스트 데이터셋 EDA (밝기 히스토그램 분석)
python main.py eda --metadata data/metadata.json --output outputs/eda_summary.json

# 5. 20장 정량적 성능 평가 + 파라미터 스윕 (파라미터 딜레마 실험)
python main.py evaluate --metadata data/metadata.json --output outputs/eval_summary.json
```

---

## 6. 8단계 실행 마일스톤

1. **Step 1 (PLAN)**: 요구사항 분해 및 `PLAN.md` 수립 (완료)
2. **Step 2 (BUILD)**: `src/` SRP 모듈 구현 + `scripts/generate_test_dataset.py`로 20장 벤치마크 생성
3. **Step 3 (TEST)**: `tests/test_scanner.py` 초고속 Mock 단위 테스트 작성 및 100% 통과 검증
4. **Step 4 (EDA, EVAL, DIAGRAMS & STUDY)**: 20장 EDA·정량 평가·파라미터 스윕 실행, `scripts/generate_diagrams_matplotlib.py`로 PNG 4종 생성, `스터디.md` 작성 (원본 100% 수록 + 상대경로 다이어그램 + 주니어 해설)
5. **Step 5 (REFINE, REPORT & VERIFIER)**: 엣지 케이스 점검, `REPORT.md`(미션 5대 분석 항목 + 엔지니어링 리포트), `scripts/verify_requirements.py`, `README.md` 완성
6. **Step 6 & 7 (2-TIER COMMENTS & NOTEBOOKLM)**: `src/*.py` 전 라인 `[1차]`/`[2차]` 2단 주석 내재화 → 테스트 재검증 → `scripts/build_notebooklm_source.py` 실행하여 `NOTEBOOKLM_FULL_SOURCE.txt` 생성 + `NOTEBOOKLM_GUIDE.md` 작성
7. **Step 8 (CORE CODE STUDY)**: `study/핵심_코드_스터디.md` 핵심 코드 8선 `[1차-단어분해]`/`[2차-예시/비유]` 작성
8. **Final Audit**: 금지 라이브러리 부재, 상대경로 검증, 전체 테스트 & 검증 스크립트 100% 통과 최종 감사
