# Battery ESS Project Guidelines (AI Assistant Instructions)

이 파일은 `Battery_ESS_Project`의 원활한 진행을 위해 AI 어시스턴트가 반드시 숙지하고 따라야 할 핵심 지침을 담고 있습니다.

## 1. 프로젝트 컨텍스트 (Project Context)
- **목표**: NASA 배터리 데이터셋을 활용한 **DeepSurv 기반 수명(EOL) 예측 모델** 개발.
- **핵심 성과**: 5-Fold Cross-Validation 기준 평균 **C-Index 0.8506** 달성 (Target > 0.7).
- **현재 상태**: 기본 모델 구현 및 검증 완료. 코드 리팩토링 및 폴더 정리(`src`, `plots`, `results`) 완료.
- **Next Steps**:
    1.  입력 데이터 고도화 (Raw Data, ICA/DVA 곡선 활용)
    2.  모델 아키텍처 진화 (CNN, LSTM 적용)

## 2. 개발 환경 및 실행 규칙 (Environment & Execution)
- **Python 실행 경로**: 반드시 아래의 가상환경 경로를 사용해야 함.
    - `& C:/Users/daeho/anaconda3/envs/EDA/python.exe`
- **파일 경로**:
    - 소스 코드: `src/` 폴더 내에 위치함. (예: `src/DeepSurv_Pytorch_v2.py`)
    - 데이터셋: `cleaned_dataset/` (절대 수정 금지)
    - 결과물: 이미지는 `plots/`, 데이터는 `results/`에 저장해야 함.
- **실행 명령어 예시**:
    - `& C:/Users/daeho/anaconda3/envs/EDA/python.exe src/visualize_model_performance.py`

## 3. 코딩 컨벤션 (Coding Standards)
- **언어**: 주석과 문서화는 반드시 **한국어**로 작성한다.
- **가독성**: 변수명은 직관적으로 짓고, 복잡한 로직에는 상세한 주석을 단다.
- **구조**:
    - 설정(Configuration) 섹션을 코드 상단에 분리하여 경로 및 하이퍼파라미터를 관리한다.
    - 메인 로직은 `if __name__ == "__main__":` 블록 안에 넣는다.

## 4. 데이터 처리 기준 (Data Processing)
- **이상치 제거**:
    - 초기 용량 < 1.5 Ah 제외
    - 총 사이클 수 < 50회 제외
- **엄선된 배터리 (Golden Samples)**:
    - `B0005`, `B0006`, `B0007`, `B0018` (NASA 표준)
    - `B0042`, `B0043`, `B0044`, `B0046`, `B0047`, `B0048` (추가 양품)
- **EOL 정의**: SOH(State of Health) < 80% 도달 시점.

## 5. 시각화 가이드라인 (Visualization)
- **그래프 스타일**:
    - 비교가 필요한 그래프는 축(Axis) 범위를 고정한다.
    - 복잡한 데이터는 Subplot을 활용하여 분리한다.
    - 제목, 축 레이블, 범례를 명확히 표시한다.
- **저장**: 생성된 그래프는 반드시 `plots/` 폴더에 저장한다.
