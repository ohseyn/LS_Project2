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

# datetime 컬럼 제거
df.drop("datetime", axis=1, inplace=True)

# SMAPE 함수 정의
def Smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# XGBoost RMSE 및 SMAPE 계산 함수
def calculate_rmse_for_num(df, num):
    num_df = df[df['num'] == num]
    X = num_df.drop(columns=['target'])
    y = num_df['target']
    train_size = len(X) - 168  # 검증 셋을 최근 168개로 설정 (24x7)
    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    smape = Smape(y_valid, y_pred)
    return rmse, smape

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

# CatBoost RMSE 및 SMAPE 계산 함수
def calculate_cat_for_num(df, num):
    num_df = df[df["num"] == num]
    X = num_df.drop(columns=["target"])
    y = num_df["target"]
    train_size = len(X) - 168
    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]
    model = CatBoostRegressor(random_state=42, verbose=0)  # CatBoost 출력 억제
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    smape = Smape(y_valid, y_pred)
    return rmse, smape

# XGBoost 결과 저장 및 출력
xgb_rmse_results = []
for num in range(1, 61):
    rmse, smape = calculate_rmse_for_num(df, num)
    xgb_rmse_results.append({'num': num, 'rmse': rmse, 'smape': smape})

for result in xgb_rmse_results:
    print(f"XGBoost - num: {result['num']}, RMSE: {result['rmse']}, SMAPE: {result['smape']}")

# LightGBM 결과 저장 및 출력
lgbm_rmse_results = []
for num in range(1, 61):
    rmse, smape = calculate_lgbm_for_num(df, num)
    lgbm_rmse_results.append({'num': num, 'rmse': rmse, 'smape': smape})

for result in lgbm_rmse_results:
    print(f"LightGBM - num: {result['num']}, RMSE: {result['rmse']}, SMAPE: {result['smape']}")

# CatBoost 결과 저장 및 출력
cat_rmse_results = []
for num in range(1, 61):
    rmse, smape = calculate_cat_for_num(df, num)
    cat_rmse_results.append({'num': num, 'rmse': rmse, 'smape': smape})

for result in cat_rmse_results:
    print(f"CatBoost - num: {result['num']}, RMSE: {result['rmse']}, SMAPE: {result['smape']}")

# 평균 RMSE 및 SMAPE 계산
xgb = pd.DataFrame(xgb_rmse_results)
xgb_rmse_mean = xgb["rmse"].mean()
xgb_smape_mean = xgb["smape"].mean()

lgbm = pd.DataFrame(lgbm_rmse_results)
lgbm_rmse_mean = lgbm["rmse"].mean()
lgbm_smape_mean = lgbm["smape"].mean()

cat = pd.DataFrame(cat_rmse_results)
cat_rmse_mean = cat["rmse"].mean()
cat_smape_mean = cat["smape"].mean()

# 평균 결과 출력
print(f"\nXGBoost 평균 RMSE: {xgb_rmse_mean}, 평균 SMAPE: {xgb_smape_mean}")
print(f"LightGBM 평균 RMSE: {lgbm_rmse_mean}, 평균 SMAPE: {lgbm_smape_mean}")
print(f"CatBoost 평균 RMSE: {cat_rmse_mean}, 평균 SMAPE: {cat_smape_mean}")
