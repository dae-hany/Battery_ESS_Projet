# DeepSurv 배터리 수명(EOL) 예측 프로젝트

이 프로젝트는 NASA 배터리 데이터셋을 활용하여 리튬 이온 배터리의 수명 종료(End-of-Life, EOL) 위험을 예측하는 DeepSurv 모델을 PyTorch로 구현한 것입니다.

## 1. 프로젝트 개요
- **목표**: 배터리 방전 데이터를 기반으로 EOL 위험도 예측 (C-Index > 0.7 목표)
- **모델**: DeepSurv (Cox Proportional Hazards Model with Deep Neural Network)
- **성과**: 5-Fold Cross-Validation 결과 **평균 C-Index 0.8506** 달성

---

## 2. 데이터 품질 분석 (EDA)

데이터 분석 및 전처리를 위해 두 가지 주요 파이썬 스크립트를 작성했습니다.

### 2.1 `eda_analysis.py` (용량 및 이상치 분석)
- **기능**:
    - 전체 배터리의 방전 용량(Capacity) 추세를 시각화합니다.
    - 초기 용량(Initial Capacity)과 사이클 수(Cycles)를 기반으로 통계를 산출합니다.
    - **이상치 탐지**: 초기 용량이 너무 낮거나(< 1.5Ah), 사이클 수가 너무 적은(< 50) 배터리를 식별합니다.
- **결과물**:
    - `capacity_trends.png`: 모든 배터리의 사이클별 용량 감소 그래프.
    - `battery_stats.csv`: 배터리별 요약 통계 (초기 용량, 최종 용량, 열화율 등).
- **분석 결과**:
    - B0005, B0006, B0007, B0018 등 표준적인 NASA 배터리는 일관된 열화 패턴을 보입니다.
    - 반면, 일부 배터리(예: B0039, B0040 등)는 초기 용량이 비정상적으로 낮거나 데이터가 부족하여 학습에서 제외하기로 결정했습니다.

### 2.2 `eda_covariates.py` (공변량 분석)
- **기능**:
    - 주요 공변량인 **전압(Voltage), 전류(Current), 온도(Temperature)**의 변화를 시각화합니다.
    - 배터리 수명 주기 내 초기, 중간, 마지막 사이클의 데이터를 비교합니다.
- **결과물**:
    - `covariates_B0005.png` 등: 특정 배터리의 사이클별 전압/전류/온도 곡선.
- **분석 결과**:
    - 정상 배터리의 경우 사이클이 진행됨에 따라 방전 시간이 짧아지고, 온도가 더 빠르게 상승하는 경향이 확인되었습니다.
    - 이는 배터리 내부 저항 증가와 용량 감소를 반영하며, 모델의 유효한 피처(Feature)로 사용될 수 있음을 시사합니다.

---

## 3. DeepSurv 모델 구현 (`DeepSurv_Pytorch_v2.py`)

핵심 모델링 코드는 `DeepSurv_Pytorch_v2.py`에 구현되어 있습니다.

### 3.1 주요 클래스 및 함수
1.  **`BatteryDataset` 클래스 (데이터 로딩 및 전처리)**
    -   **데이터 필터링**: EDA 결과를 바탕으로 이상치 배터리를 자동으로 제외합니다.
    -   **SOH 계산 및 스무딩**: 용량 데이터를 이동 평균(Rolling Mean)으로 스무딩하여 노이즈를 제거하고 SOH(State of Health)를 계산합니다.
    -   **이벤트 정의 (Censoring)**: SOH가 80% 미만으로 떨어지는 지점을 '이벤트 발생(Event=1)'으로 정의하고, 도달하지 못한 경우 '중도 절단(Censored, Event=0)'으로 처리합니다.
    -   **피처 추출**: 각 사이클에서 다음 4가지 피처를 추출하여 입력값으로 사용합니다.
        -   방전 시간 (Discharge Time)
        -   최대 온도 (Max Temperature)
        -   최소 전압 (Min Voltage)
        -   스무딩된 용량 (Smoothed Capacity)

2.  **`DeepSurv` 클래스 (모델 아키텍처)**
    -   PyTorch `nn.Module`을 상속받은 다층 퍼셉트론(MLP) 구조입니다.
    -   **구조**: Input(4) -> Linear(64) -> ReLU -> BatchNorm -> Dropout -> Linear(32) -> ... -> Output(1)
    -   출력값은 로그 위험률(Log Hazard Ratio)입니다.

3.  **`cox_ph_loss` 함수 (손실 함수)**
    -   Cox Proportional Hazards 모델의 부분 우도(Partial Likelihood)를 음의 로그 우도(Negative Log Likelihood)로 변환하여 손실 함수로 사용합니다.
    -   이벤트가 발생한 시점들의 순서를 고려하여 위험률을 최적화합니다.

4.  **`run_kfold_cv` 함수 (검증)**
    -   데이터셋을 5개의 폴드(Fold)로 나누어 교차 검증(Cross-Validation)을 수행합니다.
    -   모델의 일반화 성능을 평가하고 과적합을 방지합니다.

### 3.2 성능 평가
- **평가 지표**: C-Index (Concordance Index)
- **최종 결과**: 5-Fold CV 평균 **0.8506**
    - Fold 1: 0.7682
    - Fold 2: 0.9262
    - Fold 3: 0.7318
    - Fold 4: 0.9005
    - Fold 5: 0.9262
- **결론**: 목표치인 0.7을 크게 상회하는 우수한 예측 성능을 확보했습니다.
