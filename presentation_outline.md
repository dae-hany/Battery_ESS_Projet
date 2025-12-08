# DeepSurv 배터리 수명 예측 프로젝트 발표 자료 (PPT 구성안)

## Slide 1: 프로젝트 개요 (Project Overview)
- **주제**: DeepSurv 모델을 이용한 리튬 이온 배터리 수명(EOL) 예측
- **목표**: 배터리 방전 데이터를 분석하여 SOH(State of Health) 80% 도달 시점을 예측
- **핵심 성과**: NASA 데이터셋 활용, 5-Fold Cross-Validation 기준 평균 C-Index **0.8506** 달성

## Slide 2: 데이터 품질 분석 (EDA)
- **데이터셋**: NASA Battery Dataset (Li-ion 18650)
- **분석 방법**:
    - 전체 배터리의 사이클별 용량(Capacity) 감소 추세 시각화
    - 초기 용량(Initial Capacity) 및 총 사이클 수(Cycle Life) 분포 확인
- **주요 발견**:
    - 표준 배터리 그룹(B0005 등)은 선형적인 열화 패턴을 보임
    - 일부 배터리는 초기 용량이 비정상적으로 낮거나(< 1.5Ah), 데이터가 조기 중단됨(< 50 cycles)

## Slide 3: 데이터 전처리 및 선별 (Data Preprocessing)
- **이상치 제거 기준 (Outlier Rejection)**:
    1.  **Initial Capacity < 1.5 Ah**: 실험 조건이 다르거나 불량인 배터리 제외
    2.  **Total Cycles < 50**: 열화 패턴을 학습하기에 데이터가 불충분한 배터리 제외
- **최종 데이터셋**:
    - 총 34개 배터리 중 **10개 엄선** (B0005, B0006, B0007, B0018, B0042 등)
    - 고품질의 방전 데이터만 학습에 활용

## Slide 4: 학습 데이터 구성 (Dataset Construction)
- **입력 데이터 (Input Features)**: 각 사이클마다 4가지 핵심 특징 추출
    1.  **Discharge Time**: 방전 완료까지 걸린 시간 (초)
    2.  **Max Temperature**: 방전 중 최고 온도
    3.  **Min Voltage**: 방전 중 최저 전압
    4.  **Smoothed Capacity**: 이동 평균(Rolling Mean)으로 노이즈를 제거한 용량
- **데이터 형태 (Shape)**: `(Batch Size, 4)`
- **타겟 데이터 (Target Labels)**:
    - **Event (E)**: 1 (SOH < 80% 도달), 0 (도달 전 데이터 종료/Censored)
    - **Time (T)**: 이벤트 발생 시점(Cycle) 또는 마지막 관측 시점

## Slide 5: 모델 구조 (DeepSurv Architecture)
- **모델 개요**: Cox Proportional Hazards Model과 Deep Neural Network의 결합
- **네트워크 구조 (MLP)**:
    - **Input Layer**: 4 Nodes (Features)
    - **Hidden Layer 1**: 64 Nodes + ReLU + BatchNorm + Dropout
    - **Hidden Layer 2**: 32 Nodes + ReLU + BatchNorm + Dropout
    - **Output Layer**: 1 Node (Log Hazard Ratio)
- **특징**: 비선형적인 배터리 열화 패턴을 효과적으로 학습하기 위해 심층 신경망 사용

## Slide 6: 학습 방법 및 손실 함수 (Training Methodology)
- **손실 함수 (Loss Function)**: **CoxPH Loss** (Negative Log Partial Likelihood)
    - 이벤트가 발생한 배터리와 그렇지 않은 배터리 간의 위험률(Hazard Ratio) 순서를 최적화
- **최적화 기법**:
    - Optimizer: Adam (Learning Rate: 0.001)
    - Batch Size: 32

## Slide 7: 실험 설계 및 검증 (Experimental Setup)
- **검증 방법**: **5-Fold Cross-Validation**
    - 전체 10개 배터리를 5개 그룹으로 분할
    - 각 Fold마다 **Train: 8개 / Test: 2개** 배터리 사용
    - 특정 배터리에 과적합되지 않고 일반화된 성능을 평가하기 위함
- **평가 지표**: **C-Index (Concordance Index)**
    - 예측된 위험도 순서와 실제 수명 종료 순서의 일치 정도 (1.0에 가까울수록 좋음)

## Slide 8: 실험 결과 (Results)
- **성능 요약**:
    - Fold 1: 0.7682
    - Fold 2: 0.9262
    - Fold 3: 0.7318
    - Fold 4: 0.9005
    - Fold 5: 0.9262
- **최종 평균 C-Index**: **0.8506**
- **결론**: 목표치(0.7)를 크게 상회하며, 다양한 배터리 특성에 대해 안정적인 예측 성능 확보

## Slide 9: 결론 및 향후 과제 (Conclusion)
- **결론**:
    - 동적 피처(전압, 온도, 시간)를 활용한 DeepSurv 모델이 배터리 수명 예측에 유효함을 입증
    - 데이터 전처리를 통한 고품질 데이터 선별이 성능 향상에 기여
- **향후 과제**:
    - 더 많은 배터리 데이터 확보 및 모델 강건성 테스트
    - 전압/전류 Raw 데이터를 직접 처리하는 CNN/LSTM 기반 모델 고도화
