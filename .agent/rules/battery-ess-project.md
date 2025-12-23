---
trigger: always_on
---

# Battery ESS Project Guidelines (AI Assistant Instructions)

이 파일은 `Battery_ESS_Project`의 고도화(Advanced Phase)를 위해 AI 어시스턴트가 반드시 숙지하고 따라야 할 핵심 지침을 담고 있습니다.

## 1. 프로젝트 컨텍스트 (Project Context)
- **목표**: NASA 배터리 데이터셋의 시계열 특성을 반영하여 **LSTM 기반 DeepSurv 잔존수명(RUL) 예측 모델** 개발.
- **기존 성과**: 정적 특성(Static Features) 기반 MLP 모델로 C-Index ~0.89 달성 (Reference: `DeepSurv_Journal_Analysis.py`).
- **핵심 과제 (Advanced Phase)**:
    1. **Sliding Window**: 데이터를 시퀀스 형태 `(Batch, Window, Features)`로 변환.
    2. **LSTM 적용**: 시계열 패턴(Degradation Trend)을 스스로 학습하도록 모델 아키텍처 변경.
    3. **Hyperparameter Tuning**: Optuna를 활용한 최적화.
    4. **XAI**: SHAP 등을 활용한 시계열 입력 특성 중요도 분석.

## 2. 개발 환경 및 실행 규칙 (Environment & Execution)
- **Python 실행 경로**: 반드시 아래의 가상환경 경로를 사용해야 함.
    - `& C:/Users/daeho/anaconda3/envs/EDA/python.exe`
- **파일 경로 구조 (Strict Structure)**:
    - `src/data_loader.py`: 데이터 전처리 및 Sliding Window 데이터셋 생성 클래스.
    - `src/models.py`: LSTMDeepSurv 등 모델 아키텍처 정의.
    - `src/trainer.py`: 학습 루프, Loss 계산, 검증 로직.
    - `src/main.py`: 전체 파이프라인 실행 및 결과 저장.
    - `results/`: 모델 가중치(.pth) 및 결과 메트릭(.json, .csv) 저장.
    - `plots/`: 학습 곡선, 생존 함수, RUL 예측 시각화 이미지 저장.
- **실행 명령어 예시**:
    - `& C:/Users/daeho/anaconda3/envs/EDA/python.exe src/main.py`

## 3. 코딩 컨벤션 (Coding Standards)
- **언어**: 주석과 문서화(Docstring)는 **한국어**로 작성한다.
- **라이브러리**: PyTorch, torchtuples(필요 시), Optuna, shap, scikit-learn 등을 활용한다.
- **데이터 처리**:
    - `DeepSurv_Journal_Analysis.py`에 구현된 전처리 로직(이상치 제거, Smoothing)을 계승하되, 구조를 시계열로 변경한다.
    - `Window Size`는 하이퍼파라미터로 관리한다.

## 4. 응답 가이드라인
- 코드 작성 시, 반드시 전체 코드를 한 번에 출력하지 말고, 모듈별로 나누어 작성하거나 수정된 부분만 명확히 제시한다.
- 오류 발생 시, 단순히 에러 메시지만 해석하지 말고 데이터의 Shape(`[B, W, F]`)과 모델의 Input/Output 차원을 먼저 점검한다.