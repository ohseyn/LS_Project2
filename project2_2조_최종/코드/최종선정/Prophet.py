import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prophet import Prophet  # Prophet 라이브러리 사용

# 데이터 로드 및 전처리
df_raw = pd.read_csv("data_week2.csv", encoding="CP949")
df = df_raw.copy()

# 컬럼명 재설정 및 날짜 변환
df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index('datetime', inplace=True)

# SMAPE 함수 정의
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# 성능 저장할 리스트 초기화
rmse_list = []
smape_list = []

# 서브플롯 설정 (10행 6열의 서브플롯)
fig, axs = plt.subplots(10, 6, figsize=(30, 20))
fig.suptitle('Prophet Model Predictions for All Buildings (Test Data Only)', fontsize=20)

# 그룹별로 Prophet 모델 학습 및 예측
for idx, (num, group) in enumerate(df.groupby('num')):
    print(f"Processing Building: {num}")
    
    # 데이터를 훈련과 검증으로 나눔 (검증 데이터는 마지막 7일의 데이터)
    train = group.iloc[:len(group)-7*24]
    valid = group.iloc[len(group)-7*24:]
    
    # Prophet에 맞게 데이터 프레임 형식 변경
    train_prophet = train.reset_index()[['datetime', 'target']]
    train_prophet.columns = ['ds', 'y']  # Prophet에 맞는 컬럼명으로 변경

    # Prophet 모델 정의 및 학습
    model = Prophet()
    model.fit(train_prophet)
    
    # 예측용 데이터프레임 생성
    future = model.make_future_dataframe(periods=len(valid), freq='H')  # 테스트 데이터 길이만큼 미래 예측
    forecast = model.predict(future)
    
    # 검증 데이터와 예측값 비교
    valid_pred = forecast.iloc[-len(valid):]['yhat'].values  # 예측값
    valid_actual = valid['target'].values  # 실제값

    # 성능 지표 계산 (RMSE, SMAPE)
    rmse = np.sqrt(mean_squared_error(valid_actual, valid_pred))
    smape_value = smape(valid_actual, valid_pred)
    
    # 성능 오차 저장
    rmse_list.append(rmse)
    smape_list.append(smape_value)
    
    # 서브플롯 위치 지정
    ax = axs[idx // 6, idx % 6]
    
    # 테스트 데이터만 서브플롯에 그리기
    ax.plot(valid.index, valid['target'], label='Actual', color='green')
    ax.plot(valid.index, valid_pred, label='Prediction', linestyle='--', color='orange')
    ax.set_title(f'Building {num} | RMSE: {rmse:.4f} | SMAPE: {smape_value:.4f}%', fontsize=10)
    ax.grid(True)

# 전체 서브플롯 레이아웃 조정 및 표시
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 상단에 제목 추가 공간 확보
plt.show()

# 성능 오차의 평균 계산
rmse_avg = np.mean(rmse_list)
smape_avg = np.mean(smape_list)

# 성능 오차의 평균 출력
print(f'Average RMSE for all buildings: {rmse_avg:.4f}')
print(f'Average SMAPE for all buildings: {smape_avg:.4f}%')
