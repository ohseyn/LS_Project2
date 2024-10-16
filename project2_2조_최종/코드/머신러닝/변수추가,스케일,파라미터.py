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
train_x = train_df.drop("target", axis=1)
train_y = train_df["target"] 

valid_df = pd.concat(valid)
valid_x = valid_df.drop("target", axis=1)
valid_y = valid_df["target"] 


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


# XGBoost 하이퍼파라미터 그리드 설정
xgb_param_grid = {
    'n_estimators': [70,100],  # 기본값: 100
    'learning_rate': [0.3, 0.01, 0.1],  # 기본값: 0.3
    'subsample': [1.0, 0.7, 0.5],          # 기본값: 1.0
    'colsample_bytree': [1.0, 0.7],   # 기본값: 1.0
    'min_child_weight': [1, 3]         # 기본값: 1
}


rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# GridSearchCV 설정
xgb_model = XGBRegressor(random_state=42)

xgb_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=xgb_param_grid,  # param_distributions -> param_grid 로 수정
    scoring=rmse_scorer, 
    verbose=2,  # 출력 수준
    n_jobs=-1  # 병렬처리로 빠르게 계산
)

# 학습 데이터에 대한 하이퍼파라미터 튜닝 실행
xgb_search.fit(train_x, train_y)

# 최적 파라미터 확인
print("Best parameters found: ", xgb_search.best_params_)

# 최적 파라미터로 모델을 재학습
best_xgb_model = xgb_search.best_estimator_

# 예측
xgb_pred = best_xgb_model.predict(valid_x)



# MSE 계산
xgb_mse = mean_squared_error(valid_y, xgb_pred)


# 결과 출력
print(f"XGB RMSE = {round(np.sqrt(xgb_mse), 2)}")
print(f"XGB SMAPE = {round(smape(valid_y, xgb_pred), 2)}")
print(f"{round(np.sqrt(xgb_mse), 2)} / {round(smape(valid_y, xgb_pred), 2)}")


# LightGBM 하이퍼파라미터 그리드 설정
lgb_param_grid = {
    'n_estimators': [100, 300],           
    'max_depth': [-1, 5],                    
    'learning_rate': [0.05, 0.1, 0.15],       
    'num_leaves': [20, 31, 40],                
    'min_child_samples': [20, 40],          
    'subsample': [0.8, 1.0],                    
    'colsample_bytree': [0.8, 1.0],            
    'reg_alpha': [0, 0.1],                     
    'reg_lambda': [0.01, 0.1]                  
}

# GridSearchCV 설정
lgb_model = LGBMRegressor(random_state=42)

lgb_search = GridSearchCV(
    estimator=lgb_model, 
    param_grid=lgb_param_grid,  # param_distributions -> param_grid 로 수정
    scoring=rmse_scorer, 
    verbose=2,  
    n_jobs=-1  
)

# 학습 데이터에 대한 하이퍼파라미터 튜닝 실행
lgb_search.fit(train_x, train_y)

# 최적의 하이퍼파라미터 출력
print("Best LightGBM Parameters:", lgb_search.best_params_)

# 최적 하이퍼파라미터를 사용해 학습 및 예측
lgb_best_model = lgb_search.best_estimator_
lgb_pred = lgb_best_model.predict(valid_x)

# MSE 계산
lgb_mse = mean_squared_error(valid_y, lgb_pred)
print(f"LGB RMSE = {round(np.sqrt(lgb_mse), 2)}")
print(f"LGB SMAPE = {round(smape(valid_y, lgb_pred), 2)}")
print(f"{round(np.sqrt(lgb_mse), 2)} / {round(smape(valid_y, lgb_pred), 2)}")


# CatBoost 하이퍼파라미터 그리드
cat_param_grid = {
    'iterations': [500, 750],               
    'depth': [4, 6],                        
    'learning_rate': [0.03, 0.05]           
}

# CatBoostRegressor 모델
cat_model = CatBoostRegressor(random_state=42, verbose=0, early_stopping_rounds=50)

# GridSearchCV 설정
cat_search = GridSearchCV(
    estimator=cat_model,
    param_grid=cat_param_grid,  # param_distributions -> param_grid 로 수정
    scoring=rmse_scorer,  
    verbose=2,  
    n_jobs=-1  
)

# 하이퍼파라미터 튜닝 실행
cat_search.fit(train_x, train_y)

# 최적 파라미터 확인
print("Best parameters found: ", cat_search.best_params_)

# 최적 파라미터로 학습된 모델을 사용해 예측
cat_best_model = cat_search.best_estimator_
cat_pred= cat_best_model.predict(valid_x)


# 결과 출력
cat_mse = mean_squared_error(valid_y, cat_pred)
print(f"CatBoost RMSE = {round(np.sqrt(cat_mse), 2)}")
print(f"CatBoost SMAPE = {round(smape(valid_y,cat_pred),2)}")
print(f"{round(np.sqrt(cat_mse),2)} / {round(smape(valid_y,cat_pred),2)}")