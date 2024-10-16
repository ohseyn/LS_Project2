import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import holidays

# 데이터 로드 및 전처리
df_raw = pd.read_csv("data_week2.csv", encoding="CP949")
df = df_raw.copy()

kr_holidays = holidays.KR()

df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df['month'] = df.datetime.dt.month                    
df['day'] = df.datetime.dt.day                        
df['hour'] = df.datetime.dt.hour                      
df['weekday'] = df.datetime.dt.weekday                
df['dayofyear'] = df.datetime.dt.dayofyear            
df['holiday'] = df['datetime'].apply(lambda x: 1 if x in kr_holidays else 0)  

# 시간 특성 생성 (주기적 특성)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Rolling Mean 및 Rolling Std 추가
df['rolling_mean'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['rolling_std'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).std())

# 스케일링 적용
scaler = StandardScaler()
scaled_columns = ['temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

# 더미 변수화 (요일)
df = pd.get_dummies(df, columns=['weekday'], drop_first=True)

# 1차 차분 적용
df['target_diff'] = df.groupby('num')['target'].diff()

# NaN 값 제거 (차분 때문에 첫 번째 값은 NaN이 됩니다)
df.dropna(subset=['target_diff'], inplace=True)

# datetime 컬럼 제거
df.drop("datetime", axis=1, inplace=True)

# 학습 및 검증 데이터셋 분리 (차분 적용 후 NaN 제거한 상태에서 분리)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# 원본 valid_df 저장 (추후에 다시 사용)
valid_df_copy = valid_df.copy()

# 기존 'target' 대신 차분된 'target_diff'을 학습에 사용
train_x = train_df.drop(['target', 'target_diff'], axis=1)
train_y = train_df['target_diff']
valid_x = valid_df.drop(['target', 'target_diff'], axis=1)
valid_y = valid_df['target_diff']

# XGBoost 모델 학습
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(train_x, train_y)

xgb_pred_diff = xgb_model.predict(valid_x)

# 차분 복원 (예측한 차분값에 이전 target 값들을 더해 복원)
valid_df['target_diff_pred'] = xgb_pred_diff
valid_df['prev_target'] = valid_df.groupby('num')['target'].shift(1)

# NaN 값이 있는 행 제거 (shift로 인해 NaN 발생)
valid_df = valid_df.dropna(subset=['prev_target', 'target_diff_pred'])

# 복원된 예측값 계산
valid_df['target_pred'] = valid_df['prev_target'] + valid_df['target_diff_pred']

# 복원된 예측값으로 RMSE 계산
xgb_mse_diff = mean_squared_error(valid_df['target'], valid_df['target_pred'])
print("XGBoost RMSE after differencing:", np.sqrt(xgb_mse_diff))

# ===================================================
# LightGBM 모델 학습
valid_df_lgb = valid_df_copy.copy()  # 원본 데이터에서 복사

lgb_model = LGBMRegressor(random_state=42)
lgb_model.fit(train_x, train_y)

lgb_pred_diff = lgb_model.predict(valid_x)

# 차분 복원 (예측한 차분값에 이전 target 값들을 더해 복원)
valid_df_lgb['target_diff_pred'] = lgb_pred_diff
valid_df_lgb['prev_target'] = valid_df_lgb.groupby('num')['target'].shift(1)

# NaN 값 제거 후 복원된 예측값 계산
valid_df_lgb = valid_df_lgb.dropna(subset=['prev_target', 'target_diff_pred'])
valid_df_lgb['target_pred'] = valid_df_lgb['prev_target'] + valid_df_lgb['target_diff_pred']

# RMSE 계산
lgb_mse_diff = mean_squared_error(valid_df_lgb['target'], valid_df_lgb['target_pred'])
print("LightGBM RMSE after differencing:", np.sqrt(lgb_mse_diff))

# ===================================================
# CatBoost 모델 학습
valid_df_cat = valid_df_copy.copy()  # 원본 데이터에서 복사

cat_model = CatBoostRegressor(random_state=42, verbose=0)
cat_model.fit(train_x, train_y)

cat_pred_diff = cat_model.predict(valid_x)

# 차분 복원
valid_df_cat['target_diff_pred'] = cat_pred_diff
valid_df_cat['prev_target'] = valid_df_cat.groupby('num')['target'].shift(1)

# NaN 값 제거 후 복원된 예측값 계산
valid_df_cat = valid_df_cat.dropna(subset=['prev_target', 'target_diff_pred'])
valid_df_cat['target_pred'] = valid_df_cat['prev_target'] + valid_df_cat['target_diff_pred']

# RMSE 계산
cat_mse_diff = mean_squared_error(valid_df_cat['target'], valid_df_cat['target_pred'])
print("CatBoost RMSE after differencing:", np.sqrt(cat_mse_diff))

# ===================================================
# 앙상블 가중치 적용 (XGBoost에 더 큰 가중치 부여)
ensemble_pred_weighted = (0.1 * xgb_pred_diff + 0.6 * lgb_pred_diff + 0.3 * cat_pred_diff)

# 차분 복원 (앙상블 예측)
valid_df_ensemble = valid_df_copy.copy()  # 원본 데이터에서 복사
valid_df_ensemble['ensemble_pred_diff'] = ensemble_pred_weighted
valid_df_ensemble['prev_target'] = valid_df_ensemble.groupby('num')['target'].shift(1)

# NaN 제거 후 앙상블 복원된 예측값 계산
valid_df_ensemble = valid_df_ensemble.dropna(subset=['prev_target', 'ensemble_pred_diff'])
valid_df_ensemble['ensemble_target_pred'] = valid_df_ensemble['prev_target'] + valid_df_ensemble['ensemble_pred_diff']

# RMSE 계산
ensemble_mse_weighted = mean_squared_error(valid_df_ensemble['target'], valid_df_ensemble['ensemble_target_pred'])
print("Weighted Ensemble RMSE:", np.sqrt(ensemble_mse_weighted))