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


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# xgboost
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(train_x, train_y)

xgb_pred = xgb_model.predict(valid_x)
xgb_mse = mean_squared_error(valid_y, xgb_pred)
print(f"XGB RMSE = {round(np.sqrt(xgb_mse),2)}")
print(f"XGB SMAPE = {round(smape(valid_y,xgb_pred),2)}")

# lgbm
lgb_model = LGBMRegressor(random_state=42)
lgb_model.fit(train_x, train_y)

lgb_pred = lgb_model.predict(valid_x)
lgb_mse = mean_squared_error(valid_y, lgb_pred)
print(f"LGB RMSE = {round(np.sqrt(lgb_mse),2)}")
print(f"LGB SMAPE = {round(smape(valid_y,lgb_pred),2)}")

# catboost
cat_model = CatBoostRegressor(random_state=42, verbose=0)  # verbose=0: 학습 로그 출력하지 않음
cat_model.fit(train_x, train_y)

cat_pred = cat_model.predict(valid_x)
cat_mse = mean_squared_error(valid_y, cat_pred)
print(f"Catboost RMSE = {round(np.sqrt(cat_mse),2)}")
print(f"Catboost SMAPE = {round(smape(valid_y,cat_pred),2)}")
