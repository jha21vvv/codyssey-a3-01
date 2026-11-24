# 🎙️ NOTEBOOKLM_GUIDE.md — Google NotebookLM 학습 & Audio Overview 팟캐스트 기획서

---

## ✅ [최상단 필수] Google NotebookLM 업로드 추천 소스 파일 목록

> Google NotebookLM은 `.py` 확장자 직접 업로드를 지원하지 않습니다.  
> 아래 **체크리스트의 5개 파일**을 그대로 NotebookLM 새 노트북에 소스로 업로드하면, 코드·이론·실험 데이터·학습 주석이 완벽히 연결된 최고 품질의 팟캐스트와 Q&A를 생성할 수 있습니다.

- [x] **`NOTEBOOKLM_FULL_SOURCE.txt`** — 테스트 코드와 프롬프트 가이드를 제외한 **순수 프로덕션 소스 코드 전 파일(`main.py` + `src/*.py`, 2단 주석 포함)** 단일 텍스트 번들 (`python scripts/build_notebooklm_source.py`로 즉시 갱신 가능)
- [x] **`스터디.md`** — 원본 과제 전문 100% 수록, 1:1 요구사항 체크리스트, 모듈별 일상 비유, 5종 다이어그램 해설
- [x] **`REPORT.md`** — 20장 정량적 성능 평가표, 실패 원인 6건 심층 분석, 파라미터 딜레마 실증 데이터, 규칙 기반 한계 및 딥러닝 가설, 엔지니어링 엣지케이스 리포트
- [x] **`study/핵심_코드_스터디.md`** — 실무·면접 재사용 빈도가 높은 핵심 코드 8선의 `[1차-단어분해]` + `[2차-예시/비유]` 정밀 해설서
- [x] **`PLAN.md`** — SRP 모듈 설계 원칙, 제약 조건(Zero DL/OCR/Scipy), 예외 처리 규격서

---

## 1. 팟캐스트 콘셉트 & 호스트 2인 대화 유도 스토리

- **에피소드 제목**: *"규칙 기반 문서 스캐너를 직접 짜보며 깨달은 것 — 왜 딥러닝이 필요할까?"*
- **호스트 A (시니어 컴퓨터 비전 아키텍트)**: OpenCV/NumPy 파이프라인의 수학적 원리(컨볼루션, Homography 행렬, 45° 마름모 편각 정렬, Sobel 기울기 압축)와 엔지니어링 트레이드오프를 날카롭게 짚어줍니다.
- **호스트 B (3개월 차 주니어 멘토)**: 청취자 눈높이에서 *"아니, 그냥 Canny 임계값만 낮추면 그림자 문서도 다 잡히는 거 아닌가요?"* 같은 핵심 질문을 던지고, 일상 비유로 개념을 정리해 줍니다.

### 핵심 대화 질문 리스트 (Discussion Flow)
1. **오프닝 (왜 한글 경로에서 OpenCV가 죽을까?)**: Windows `C:\Users\안재현\...` 폴더에서 `cv2.imread`가 조용히 `None`을 뱉는 함정과, `np.fromfile` + `cv2.imdecode` 우회 패턴.
2. **기하학 하이라이트 (45도 마름모의 반란)**: 블로그에 흔한 `argmin(x+y)` 4점 정렬이 45도 회전 문서에서 중복 꼭짓점을 뽑아 망가지는 이유, 그리고 `scipy`/`shapely` 없이 무게중심 편각(`atan2`)으로 완벽히 방어한 방법.
3. **데이터로 본 어둠의 과학 (EDA 히스토그램)**: 왜 쉬운 사진은 양봉(Bimodal) 분포를 보이고, 어두운 그림자 사진(`shadow_04`, 평균 밝기 20.36)은 기울기 `|∇I|`가 `13.55`로 추락해 엣지가 사라지는가?
4. **클라이맥스 (파라미터 딜레마)**: `complex_02`(격자 배경)를 살리는 `strong_denoise`와 `shadow_04`(암부 그림자)를 살리는 `morph_bridge`가 정면충돌하고, `tilt_04`를 살리는 `sensitive_shadow`가 오히려 쉬운 `easy_02`를 망가뜨리는 두더지 잡기(Whack-a-Mole) 실화!
5. **피날레 (규칙 기반에서 딥러닝으로)**: 손가락 가림(`complex_05`)이나 빛 반사(`shadow_05`)처럼 5가지 규칙 조합으로도 절대 못 푸는 5장(25%)을 U-Net 세그멘테이션과 Corner Heatmap 딥러닝 모델은 어떻게 해결하는가?

---

## 2. 핵심 기술적 결정 3가지 & 재미있는 일상 비유 모음

| 핵심 기술적 결정 | 일상생활 비유 | 팟캐스트 강조 포인트 |
| :--- | :--- | :--- |
| **1. 무게중심 편각(`atan2`) + 사전순 타이브레이킹 4점 정렬** | **"원탁에 둘러앉은 4명에게 시계 방향으로 번호표 붙이기"** | 단순 `x+y` 방식은 45도 마름모에서 두 사람이 동시에 "내가 1등!"이라며 겹치지만, 중심에 서서 시계 바늘처럼 360도 돌며 번호를 매기면 절대 겹치지 않습니다. |
| **2. 적응형 이진화(`ADAPTIVE_THRESH_GAUSSIAN_C`)** | **"동네마다 밝기를 따로 재는 맞춤형 복사기"** | 전국 공통 합격선(전역 임계값 128)을 쓰면 그림자 동네는 전부 새까맣게 탈락하지만, 15×15 동네 평균을 따로 구하면 어두운 쪽 글씨도 살려냅니다(단, 너무 날카로운 그림자 선은 예외!). |
| **3. 5개 프리셋 교차 파라미터 스윕(Parameter Dilemma)** | **"안경 도수 하나로 근시·원시·난시를 다 고치려는 딜레마"** | 큰 블러(`strong_denoise`)는 시끄러운 격자 배경엔 명약이지만 어두운 그림자엔 독약입니다. 단일 고정 규칙의 천장(65%)을 넘어서는 열쇠가 바로 데이터 기반 학습입니다. |

---

## 3. 3개월 차 주니어가 반드시 깨달아야 할 청취 포인트 3가지

1. **이미지는 마법이 아니라 `(Height, Width, Channel)` 3차원 NumPy 숫자 배열이다**: 흑백 변환·블러(컨볼루션 내적)·투시 변환(3×3 행렬 곱셈)·마스크 IoU 모두 데이터 분석에서 배운 NumPy 배열 연산의 연장선입니다.
2. **예외 방어와 재현성이 실무 코드의 품질을 가른다**: Windows 한글 경로 `imdecode`, 짝수 커널 자동 홀수 정규화, 45° 동률 방어, Mock 기반 0.3초 단위 테스트처럼 "내가 놓치기 쉬운 엣지 케이스"를 코드로 막아두는 습관이 주니어와 시니어를 가릅니다.
3. **딥러닝을 배우는 진짜 이유를 숫자로 이해했다**: "딥러닝이 요즘 유행이라서"가 아니라, *직접 규칙 기반 OpenCV 파이프라인을 구현해 보니 20장 중 9장(45%)→최대 13장(65%)에서 파라미터 충돌과 가림/반사의 벽에 부딪혔기 때문*에 데이터로부터 의미론적 특징을 학습하는 방식이 필수적임을 체감하게 됩니다.

---

## 4. NotebookLM Customize Audio Overview 전용 프롬프트 (복사용)

### 🇰🇷 한국어 전용 프롬프트
```text
두 명의 호스트(시니어 컴퓨터 비전 아키텍트와 3개월 차 주니어 개발자 멘토)가 대화하는 15분 내외의 한국어 딥다이브 팟캐스트를 진행해 주세요.
1. 반드시 업로드된 실제 실험 수치를 정확히 인용하세요: 기본 설정 성공률 45%(9/20, 단순+정면 100%, 그림자 40%, 45도+ 40%, 복잡한 배경 0%), 단일 최고 프리셋 65%(13/20), Oracle 합집합 75%(15/20), 그리고 전 프리셋 실패 5건(shadow_03, shadow_05, complex_03, complex_04, complex_05).
2. 특히 '45도 마름모에서 기존 sum/diff 좌표 정렬이 붕괴하고 무게중심 atan2 편각 정렬로 방어한 이야기', 'Windows 한글 경로 np.fromfile 우회', 'complex_02(strong_denoise 성공) vs shadow_04(morph_bridge 성공) vs tilt_04(sensitive_shadow 성공)의 파라미터 딜레마 충돌'을 생생한 일상 비유와 함께 다뤄 주세요.
3. 마지막은 "그래서 왜 우리가 다음 단계에서 딥러닝(U-Net 분할, Corner Heatmap)을 배워야 하는가?"라는 통찰로 감동적으로 마무리해 주세요.
```

### 🇺🇸 English Custom Prompt
```text
Host a 12-to-15-minute engaging deep-dive podcast between a Senior Computer Vision Architect and a Junior Developer Mentor discussing this OpenCV & NumPy Document Scanner project.
1. Ground every technical claim in the exact benchmark data from REPORT.md: 45% (9/20) default accuracy (100% easy, 40% shadow, 40% >=45° tilt, 0% complex background), 65% (13/20) best single preset ceiling, 75% (15/20) oracle union, and 5 unsolvable hard-core failures.
2. Highlight three standout engineering lessons using vivid everyday analogies: (a) why classic sum/diff 4-point ordering breaks on 45-degree rotated diamonds and how centroid polar-angle sorting with lexicographical tie-breaking fixes it without scipy/shapely; (b) Unicode-safe Windows image I/O via np.fromfile + cv2.imdecode; and (c) the empirical Parameter Dilemma proving why strong_denoise, morph_bridge, and sensitive_shadow directly contradict each other across images.
3. Conclude by connecting these firsthand rule-based limitations to why data-driven deep learning (semantic segmentation & corner heatmap regression) is essential for real-world generalization.
```
