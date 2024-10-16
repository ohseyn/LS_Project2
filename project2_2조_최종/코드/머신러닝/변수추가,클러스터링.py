import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.metrics import roc_auc_score, precision_score,recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import holidays
 


df_raw = pd.read_csv("data_week2.csv", encoding ="cp949")

df = df_raw.copy()
df.describe()
df.columns
df.head(10)
kr_holidays = holidays.KR()
df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df['month'] = df.datetime.dt.month
df['day'] = df.datetime.dt.day
df['hour'] = df.datetime.dt.hour
df['weekday'] = df.datetime.dt.weekday
df['dayofyear'] = df.datetime.dt.dayofyear
df['holiday'] = df['datetime'].apply(lambda x: 1 if x in kr_holidays else 0)  # 공휴일

# 시간 특성 생성 (주기적 특성)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Rolling Mean 및 Rolling Std 추가
df['rolling_mean'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['rolling_std'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).std())


eda_df = df.copy()
df["target"].describe()
df["target"].hist()

df["temp"].describe()
df["temp"].hist()

df["wind"].describe()
df["wind"].hist()

df["humid"].describe()
df["humid"].hist()

df["rain"].describe()
df["rain"].hist()

df.drop("datetime",axis=1,inplace=True)

df[df["target"] == 0]



zero_indices = df[df["target"] == 0].index
df["target"] = np.log(df["target"]+1)


by_weekday = df.groupby(['num','weekday'])['target'].median().reset_index().pivot(index='num', columns='weekday', values='target').reset_index()
by_hour = df.groupby(['num','hour'])['target'].median().reset_index().pivot(index='num', columns='hour', values='target').reset_index().drop('num', axis=1)
clus_df = pd.concat([by_weekday, by_hour], axis= 1)
columns = ['num'] + ['day'+str(i) for i in range(7)] + ['hour'+str(i) for i in range(24)]
clus_df.columns = columns

for i in range(len(clus_df)):
    # 요일 별 전력 중앙값에 대해 scaling
    clus_df.iloc[i,1:8] = (clus_df.iloc[i,1:8] - clus_df.iloc[i,1:8].mean())/clus_df.iloc[i,1:8].std()
    # 시간대별 전력 중앙값에 대해 scaling
    clus_df.iloc[i,8:] = (clus_df.iloc[i,8:] - clus_df.iloc[i,8:].mean())/clus_df.iloc[i,8:].std()


# 클러스터링
def change_n_clusters(n_clusters, data):
    sum_of_squared_distance = []
    for n_cluster in n_clusters:
        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(data)
        sum_of_squared_distance.append(kmeans.inertia_)
    
change_n_clusters([2,3,4,5,6,7,8,9,10,11], clus_df.iloc[:,1:])

kmeans = KMeans(n_clusters=8, random_state = 42)
km_cluster = kmeans.fit_predict(clus_df.iloc[:,1:])

df['km_cluster'] = km_cluster.repeat(122400/60)



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



def Smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


##### xgboost
def xgb(train_x, train_y, valid_x, valid_y):
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(train_x, train_y) 

    # Predictions in log space
    xgb_pred = xgb_model.predict(valid_x)
    # Reverse log transformation
    xgb_pred_original = np.exp(xgb_pred) - 1
    valid_y_original = np.exp(valid_y) - 1

    # RMSE on original scale
    xgb_rmse = np.sqrt(mean_squared_error(valid_y_original, xgb_pred_original))
    xgb_smape = Smape(valid_y_original,xgb_pred_original)
    return xgb_rmse, xgb_smape

#####lgbm
def lgbm(train_x, train_y, valid_x, valid_y):
    lgb_model = LGBMRegressor(random_state=42)
    lgb_model.fit(train_x, train_y)

    # Predictions in log space
    lgb_pred = lgb_model.predict(valid_x)
    # Reverse log transformation
    lgb_pred_original = np.exp(lgb_pred) - 1
    valid_y_original = np.exp(valid_y) - 1

    # RMSE on original scale
    lgb_rmse = np.sqrt(mean_squared_error(valid_y_original, lgb_pred_original))
    lgb_smape = Smape(valid_y_original,lgb_pred_original)
    return lgb_rmse , lgb_smape

#####catboost
def cat(train_x, train_y, valid_x, valid_y):
    cat_model = CatBoostRegressor(random_state=42, verbose=0)
    cat_model.fit(train_x, train_y)

    # Predictions in log space
    cat_pred = cat_model.predict(valid_x)
    # Reverse log transformation
    cat_pred_original = np.exp(cat_pred) - 1
    valid_y_original = np.exp(valid_y) - 1

    # RMSE on original scale
    cat_rmse = np.sqrt(mean_squared_error(valid_y_original, cat_pred_original))
    cat_smape = Smape(valid_y_original,cat_pred_original)
    return cat_rmse, cat_smape



result = []
for x in range(len(df['km_cluster'].unique())):
    # 클러스터별 훈련 데이터 가져오기
    df_tmp = df.loc[df["km_cluster"] == x, :]
    train_x_tmp = train_x.loc[train_x["km_cluster"] == x, :]
    train_y_tmp = train_y.loc[train_x_tmp.index]
    
    # 클러스터별 검증 데이터 가져오기
    valid_x_tmp = valid_x.loc[valid_x["km_cluster"] == x, :]
    valid_y_tmp = valid_y.loc[valid_x_tmp.index]

    # 데이터가 있는지 확인

    weight = len(df_tmp) / len(df)
    
    # 모델 학습 및 평가
    xgb_rmse,xgb_smape = xgb(train_x_tmp, train_y_tmp, valid_x_tmp, valid_y_tmp)
    lgbm_rmse,lgbm_smape = lgbm(train_x_tmp, train_y_tmp, valid_x_tmp, valid_y_tmp)
    cat_rmse,cat_smape = cat(train_x_tmp, train_y_tmp, valid_x_tmp, valid_y_tmp)
    
    # 결과 저장
    result.append({
        "cluster": x,
        "xgb_rmse": xgb_rmse,
        "xgb_smape": xgb_smape,
        "lgbm_rmse": lgbm_rmse,
        "lgbm_smape": lgbm_smape,
        "catboost_rmse": cat_rmse,
        "catboost_smape": cat_smape,
        "weight": weight
    })
    
weighted_rmse_xgb = sum([r['xgb_rmse'] * r['weight'] for r in result])
weighted_smape_xgb = sum([r['xgb_smape'] * r['weight'] for r in result])

weighted_rmse_lgbm = sum([r['lgbm_rmse'] * r['weight'] for r in result])
weighted_smape_lgbm = sum([r['lgbm_smape'] * r['weight'] for r in result])

weighted_rmse_catboost = sum([r['catboost_rmse'] * r['weight'] for r in result])
weighted_smape_catboost = sum([r['catboost_smape'] * r['weight'] for r in result])


print(round(weighted_rmse_xgb,2)," / ",round(weighted_smape_xgb,2))
print(round(weighted_rmse_lgbm,2)," / ",round(weighted_smape_lgbm,2))
print(round(weighted_rmse_catboost,2)," / ",round(weighted_smape_catboost,2))