import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
import holidays
from sklearn.model_selection import GridSearchCV

# 데이터 로드
df_raw = pd.read_csv("data_week2.csv", encoding="CP949")
df = df_raw.copy()

# 한국 공휴일 설정
kr_holidays = holidays.KR()

# 컬럼명 수정
df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])

# 새로운 특성 추가
df['month'] = df.datetime.dt.month                    # 월(숫자)
df['day'] = df.datetime.dt.day                        # 일(숫자)
df['hour'] = df.datetime.dt.hour                      # 시(숫자)
df['weekday'] = df.datetime.dt.weekday                # 요일(숫자)
df['dayofyear'] = df.datetime.dt.dayofyear            # 365일 중 몇 번째 날인지
df['holiday'] = df['datetime'].apply(lambda x: 1 if x in kr_holidays else 0)  # 공휴일

# 시간 특성 생성 (주기적 특성)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Rolling Mean 및 Rolling Std 추가
df['rolling_mean'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['rolling_std'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).std())

# datetime 컬럼 제거
df.drop("datetime", axis=1, inplace=True)

# SMAPE 함수 정의
def Smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# LightGBM RMSE 및 SMAPE 계산 함수
def calculate_lgbm_for_num(df, num):
    num_df = df[df["num"] == num]
    X = num_df.drop(columns=["target"])
    y = num_df["target"]
    train_size = len(X) - 168
    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    smape = Smape(y_valid, y_pred)

    return rmse, smape

# LightGBM 결과 저장 및 출력
lgbm_rmse_results = []
for num in range(1, 61):
    rmse, smape = calculate_lgbm_for_num(df, num)
    lgbm_rmse_results.append({'num': num, 'rmse': rmse, 'smape': smape})

for result in lgbm_rmse_results:
    print(f"LightGBM - num: {result['num']}, RMSE: {result['rmse']}, SMAPE: {result['smape']}")

lgbm = pd.DataFrame(lgbm_rmse_results)
lgbm_rmse_mean = lgbm["rmse"].mean()
lgbm_smape_mean = lgbm["smape"].mean()


# 평균 결과 출력
print(f"LightGBM 평균 RMSE: {lgbm_rmse_mean}, 평균 SMAPE: {lgbm_smape_mean}")
