import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score,recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import holidays
from sklearn.model_selection import RandomizedSearchCV

df_raw = pd.read_csv("data_week2.csv", encoding="CP949")

df = df_raw.copy()

kr_holidays = holidays.KR()

df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df['month'] = df.datetime.dt.month                    # 월(숫자)
df['day'] = df.datetime.dt.day                        # 일(숫자)
df['hour'] = df.datetime.dt.hour                      # 시(숫자)
df['weekday'] = df.datetime.dt.weekday                # 요일(숫자)
df['dayofyear'] = df.datetime.dt.dayofyear            # 365일 중 얼마에 해당하는지
df['holiday'] = df['datetime'].apply(lambda x: 1 if x in kr_holidays else 0)  # 공휴일

# 시간 특성 생성 (주기적 특성)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Rolling Mean 및 Rolling Std 추가
df['rolling_mean'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['rolling_std'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).std())

scaler = StandardScaler()
scaled_columns = ['temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

df = pd.get_dummies(df, columns=['weekday'], drop_first=True)

df.drop("datetime", axis=1, inplace=True)

# 학습 및 검증 데이터셋 분리
train = []
valid = []
for num, group in df.groupby('num'):
    train.append(group.iloc[:len(group)-7*24])  
    valid.append(group.iloc[len(group)-7*24:]) 

train_df = pd.concat(train)
train_x = train_df.drop("target",axis=1)
train_y = train_df["target"] 

valid_df = pd.concat(valid)
valid_x = valid_df.drop("target",axis=1)
valid_y = valid_df["target"] 

# xgboost
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(train_x, train_y)

xgb_pred = xgb_model.predict(valid_x)
xgb_mse = mean_squared_error(valid_y, xgb_pred)
print(np.sqrt(xgb_mse))

# lgbm
lgb_model = LGBMRegressor(random_state=42)
lgb_model.fit(train_x, train_y)

lgb_pred = lgb_model.predict(valid_x)
lgb_mse = mean_squared_error(valid_y, lgb_pred)
print(np.sqrt(lgb_mse))

# catboost
cat_model = CatBoostRegressor(random_state=42, verbose=0)  # verbose=0: 학습 로그 출력하지 않음
cat_model.fit(train_x, train_y)

cat_pred = cat_model.predict(valid_x)
cat_mse = mean_squared_error(valid_y, cat_pred)
print("Catboost RMSE:", np.sqrt(cat_mse))
#홀리데이 변수 추가, 시간 특성 추가(주기 데이터), 롤링 mean, 스탠다스 스케일러, 파라미터 튜닝
#세 모델 앙상블

#=======================================================
# XGBoost 하이퍼파라미터 그리드 설정
xgb_param_grid = {
    'n_estimators': [100, 200, 300, 500],  # 트리 개수
    'max_depth': [3, 5, 7, 9],             # 트리 최대 깊이
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 학습률
    'subsample': [0.6, 0.8, 1.0],          # 샘플링 비율
    'colsample_bytree': [0.6, 0.8, 1.0],   # 트리별로 사용할 feature의 비율
    'min_child_weight': [1, 3, 5],         # 과적합 방지, 잎사귀 노드의 최소 가중치
    'gamma': [0, 0.1, 0.3, 0.5],           # 과적합 방지를 위한 리프 노드의 분할에 필요한 최소 손실 감소 값
}

# RandomizedSearchCV 설정
xgb_model = XGBRegressor(random_state=42)

xgb_search = RandomizedSearchCV(
    estimator=xgb_model, 
    param_distributions=xgb_param_grid, 
    n_iter=50,  # 시도할 조합의 개수
    scoring='neg_mean_squared_error', 
    cv=3,  # 교차 검증
    verbose=2,  # 출력 수준
    random_state=42, 
    n_jobs=-1  # 병렬처리로 빠르게 계산
)

# 학습 데이터에 대한 하이퍼파라미터 튜닝 실행
# 로그 변환된 target으로 학습 진행
train_y_log = np.log(train_y + 1)
valid_y_log = np.log(valid_y + 1)

# 모델 학습 (최적 하이퍼파라미터 적용)
xgb_model.fit(train_x, train_y_log)

# 예측
xgb_pred_log = xgb_model.predict(valid_x)

# 로그 역변환 (예측 결과를 다시 원래 스케일로 변환)
xgb_pred = np.exp(xgb_pred_log) - 1

# MSE 계산
xgb_mse_tuned = mean_squared_error(valid_y, xgb_pred)
print("Tuned XGBoost RMSE after log transformation:", np.sqrt(xgb_mse_tuned))

# LightGBM 하이퍼파라미터 그리드 설정
lgb_param_grid = {
    'n_estimators': [100, 200, 300, 500],       # 트리 개수
    'max_depth': [3, 5, 7, -1],                 # 트리 최대 깊이 (-1은 무제한 깊이)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],    # 학습률
    'num_leaves': [20, 31, 40, 50],             # 트리 리프 노드 수
    'min_child_samples': [10, 20, 30, 50],      # 리프 노드가 가질 수 있는 최소 데이터 샘플 수
    'subsample': [0.6, 0.8, 1.0],               # 샘플링 비율
    'colsample_bytree': [0.6, 0.8, 1.0],        # 각 트리별로 사용할 feature의 비율
    'reg_alpha': [0, 0.01, 0.1, 1.0],           # L1 정규화 (과적합 방지)
    'reg_lambda': [0, 0.01, 0.1, 1.0]           # L2 정규화 (과적합 방지)
}

# RandomizedSearchCV 설정
lgb_model = LGBMRegressor(random_state=42)

lgb_search = RandomizedSearchCV(
    estimator=lgb_model, 
    param_distributions=lgb_param_grid, 
    n_iter=50,  # 시도할 조합의 개수
    scoring='neg_mean_squared_error', 
    cv=3,  # 교차 검증
    verbose=2,  # 출력 수준
    random_state=42, 
    n_jobs=-1  # 병렬처리로 빠르게 계산
)

# 학습 데이터에 대한 하이퍼파라미터 튜닝 실행
lgb_search.fit(train_x, train_y)

# 최적의 하이퍼파라미터 출력
print("Best LightGBM Parameters:", lgb_search.best_params_)

# 최적 하이퍼파라미터를 사용해 학습 및 예측
lgb_best_model = lgb_search.best_estimator_
lgb_pred = lgb_best_model.predict(valid_x)

# MSE 계산
lgb_mse_tuned = mean_squared_error(valid_y, lgb_pred)
print("Tuned LightGBM RMSE:", np.sqrt(lgb_mse_tuned))

#===================================================
# 앙상블 가중치 적용 (XGBoost에 더 큰 가중치 부여)
ensemble_pred_weighted = (0.1 * xgb_pred + 0.6 * lgb_pred + 0.3 * cat_pred)

# MSE 계산
ensemble_mse_weighted = mean_squared_error(valid_y, ensemble_pred_weighted)
print("Weighted Ensemble RMSE:", np.sqrt(ensemble_mse_weighted))