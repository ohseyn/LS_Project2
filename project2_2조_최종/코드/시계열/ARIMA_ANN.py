import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import holidays

kr_holidays = holidays.KR()

# 데이터 로드
df_raw = pd.read_csv("data_week2.csv", encoding="CP949")

df = df_raw.copy()

df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df['month'] = df.datetime.dt.month  # 월(숫자)
df['day'] = df.datetime.dt.day  # 일(숫자)
df['hour'] = df.datetime.dt.hour  # 시(숫자)
df['weekday'] = df.datetime.dt.weekday  # 요일(숫자)
df['holiday'] = df['datetime'].apply(lambda x: 1 if x in kr_holidays else 0)  # 공휴일
df['week'] = (df['datetime'] - df['datetime'].min()).dt.days // 7 + 1
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# 전력 소비량 데이터 추출
power_data = df['target']

# Train-validation split (ARIMA는 순서가 중요하므로 끝에서 7일 데이터를 테스트 데이터로 사용)
train_data = power_data[:-7*24]  # 마지막 7일(168시간)을 테스트 데이터로 사용
test_data = power_data[-7*24:]

# 1. ARIMA 모델 학습
arima_model = ARIMA(train_data, order=(1, 1, 1))
arima_fit = arima_model.fit()

# ARIMA 예측
arima_pred = arima_fit.forecast(steps=len(test_data))

# ARIMA 모델의 잔차 계산 (ARIMA 모델의 예측 오차, 즉 학습하지 못한 부분)
residuals = test_data - arima_pred

# 잔차 데이터 확인
print(f"잔차 데이터 길이: {len(residuals)}")
print(f"잔차 데이터 NaN 값 개수: {residuals.isna().sum()}")
print(f"잔차 데이터 샘플: {residuals.head()}")

# 잔차에서 발생하는 NaN 값 제거
residuals = residuals.dropna()

# 잔차 데이터 길이 확인 후 시차 설정
print(f"잔차 데이터 길이 (NaN 제거 후): {len(residuals)}")

# 시차 설정 (잔차 데이터 길이에 맞춰 자동 조정)
lags = 3
if len(residuals) <= lags:
    lags = max(1, len(residuals) - 1)
    print(f"잔차 데이터가 부족하여 시차를 {lags}로 자동 조정했습니다.")

def create_lagged_features(data, lags):
    X, y = [], []
    for i in range(len(data) - lags):
        X.append(data[i:i+lags])
        y.append(data[i+lags])
    return np.array(X), np.array(y)

# 학습 데이터 생성 (잔차로 ANN 학습)
train_residuals, train_residuals_target = create_lagged_features(residuals.values, lags)

# 학습할 데이터가 없는 경우 예외 발생
if train_residuals.shape[0] == 0 or train_residuals_target.shape[0] == 0:
    raise ValueError("잔차 데이터에서 충분한 샘플을 생성하지 못했습니다. 시차(Lags)를 줄이거나 데이터를 확인하세요.")

# 잔차 데이터 차원 확인
print(f"train_residuals shape: {train_residuals.shape}")
print(f"train_residuals_target shape: {train_residuals_target.shape}")

# ANN 모델 정의 (MLPRegressor는 2D 데이터를 필요로 함)
if train_residuals.ndim == 1:
    train_residuals = train_residuals.reshape(-1, lags)  # 2D로 변환
if train_residuals_target.ndim == 1:
    train_residuals_target = train_residuals_target.reshape(-1, 1)  # 2D로 변환

# ANN 모델 정의
ann_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)

# ANN 모델 학습
ann_model.fit(train_residuals, train_residuals_target)

# ANN 예측 수행
ann_pred = ann_model.predict(train_residuals)

# 3. ARIMA + ANN 결합하여 최종 예측
final_pred = arima_pred[lags:] + ann_pred

# 4. 성능 평가 (MSE, RMSE, SMAPE)
mse = mean_squared_error(test_data[lags:], final_pred)
rmse = np.sqrt(mse)

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

smap_value = smape(test_data[lags:], final_pred)

# 결과 출력
print(f'RMSE: {rmse}')
print(f'SMAPE: {smap_value}%')

# 최종 예측 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(test_data.index[lags:], test_data[lags:], label='Actual')
plt.plot(test_data.index[lags:], final_pred, label='Hybrid Model (ARIMA + ANN)', linestyle='--')
plt.legend()
plt.title('Hybrid Model (ARIMA + ANN) Predictions vs Actual')
plt.xlabel('Date')
plt.ylabel('Power Consumption')
plt.grid(True)
plt.show()
