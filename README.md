# DeepSurv 배터리 수명(EOL) 예측 프로젝트

이 프로젝트는 NASA 배터리 데이터셋을 활용하여 리튬 이온 배터리의 수명 종료(End-of-Life, EOL) 위험을 예측하는 DeepSurv 모델을 PyTorch로 구현한 것입니다.

## 1. 프로젝트 구조 (Project Structure)

```
Battery_ESS_Project/
├── src/                        # 소스 코드 (Source Code)
│   ├── DeepSurv_Pytorch_v2.py  # DeepSurv 모델 및 학습 메인 스크립트
│   ├── eda_analysis.py         # 전체 데이터 EDA 및 이상치 분석
│   ├── eda_covariates.py       # 전압, 전류, 온도 공변량 분석
│   ├── eda_selected_batteries.py # 엄선된 10개 배터리 시각화
│   ├── visualize_training_data.py # 학습 데이터 구조 시각화
│   └── visualize_model_performance.py # 모델 성능 시각화
├── plots/                      # 시각화 결과물 (Images)
│   ├── capacity_trends_grouped.png
│   ├── capacity_trends_selected.png
│   ├── training_data_visualization.png
│   └── model_performance_metrics.png
├── results/                    # 분석 결과 데이터 (CSV)
│   └── battery_stats.csv
└── cleaned_dataset/            # 데이터셋 (Dataset)
    ├── metadata.csv
    └── data/                   # 개별 사이클 데이터 파일
```

---

## 2. 데이터 품질 분석 (EDA)

### 2.1 전체 배터리 용량 추세
`src/eda_analysis.py`를 통해 전체 배터리의 용량 감소 패턴을 분석했습니다.

![](/plots/capacity_trends_grouped.png)

*   **분석**:
    *   대부분의 배터리가 사이클이 진행됨에 따라 용량이 감소하는 경향을 보입니다.
    *   일부 배터리(Group 3, 4의 하단)는 초기 용량이 1.0Ah 미만으로 매우 낮거나, 데이터가 조기에 중단되어 학습에 부적합함을 확인했습니다.
    *   이를 통해 **초기 용량 1.5Ah 이상, 사이클 50회 이상**인 배터리만 선별하는 기준을 수립했습니다.

### 2.2 엄선된 학습 데이터
`src/eda_selected_batteries.py`를 통해 학습에 사용될 고품질 배터리 10개를 선별했습니다.

![](/plots/capacity_trends_selected.png)

*   **선별된 배터리**: `B0005`, `B0006`, `B0007`, `B0018` (NASA 표준) 및 추가 양품 6개.
*   **특징**:
    *   **일관된 시작점**: 모든 배터리가 약 1.8Ah 부근에서 시작합니다.
    *   **충분한 데이터**: 최소 70 사이클 이상의 긴 수명 데이터를 보유하고 있습니다.
    *   **안정적 열화**: 급격한 노이즈 없이 완만하게 용량이 감소하여 예측 모델 학습에 이상적입니다.

---

## 3. 학습 데이터 구성 (Dataset Construction)

`src/visualize_training_data.py`를 통해 모델에 입력되는 데이터의 형태를 시각화했습니다.

![](/plots/training_data_visualization.png)

*   **입력 피처 (상단 4개 그래프)**:
    1.  **Discharge Time**: 배터리가 노화될수록 방전 시간이 짧아집니다.
    2.  **Max Temp**: 내부 저항 증가로 인해 방전 중 최고 온도가 상승합니다.
    3.  **Min Voltage**: 전압 강하(IR Drop)로 인해 최저 전압이 낮아지는 경향이 있습니다.
    4.  **Smoothed Capacity**: 현재의 건강 상태(SOH)를 나타내는 핵심 지표입니다.
*   **타겟 데이터 (하단 그래프)**:
    *   **SOH (State of Health)**: 용량이 초기 대비 80% 밑으로 떨어지는 시점을 **고장(Event=1)**으로 정의합니다.
    *   **Time**: 고장 시점까지 남은 사이클 수를 예측 목표(Target)로 합니다.

---

## 4. 모델 성능 (Model Performance)

`src/visualize_model_performance.py`를 통해 5-Fold Cross-Validation 결과를 분석했습니다.

![](/plots/model_performance_metrics.png)

### 4.1 Fold별 성능 (왼쪽 그래프)
*   **평균 C-Index**: **0.8506** (목표치 0.7 초과 달성)
*   **해석**: 5번의 교차 검증 중 4번에서 0.76 이상의 매우 높은 점수를 기록했습니다. 이는 모델이 학습 데이터에 과적합되지 않고, 새로운 배터리에 대해서도 안정적으로 수명 순서를 예측함을 의미합니다.

### 4.2 예측력 분석 (오른쪽 그래프)
*   **Risk Score vs Time**: 모델이 예측한 위험 점수(X축)와 실제 남은 수명(Y축)의 관계입니다.
*   **해석**:
    *   각 배터리 그룹(세로로 긴 점들의 집합) 내에서, **위험 점수가 높아질수록(오른쪽 이동) 남은 수명이 줄어드는(아래로 이동)** 뚜렷한 경향을 보입니다.
    *   이는 모델이 배터리가 늙어가는 과정을 정확하게 추적하고 있음을 시사합니다.

---

## 5. 실행 방법 (How to Run)

모든 스크립트는 `src` 폴더 내에 위치합니다. 프로젝트 루트에서 다음 명령어로 실행하세요.

```bash
# 1. EDA 및 이상치 분석
& C:/Users/daeho/anaconda3/envs/EDA/python.exe src/eda_analysis.py

# 2. DeepSurv 모델 학습 및 검증
& C:/Users/daeho/anaconda3/envs/EDA/python.exe src/DeepSurv_Pytorch_v2.py

# 3. 결과 시각화
& C:/Users/daeho/anaconda3/envs/EDA/python.exe src/visualize_model_performance.py
```
