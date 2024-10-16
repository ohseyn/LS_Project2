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

df.columns

df.head()

df.describe()
df["target"].mode()

kr_holidays = holidays.KR()

df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df['month'] = df.datetime.dt.month                    # 월(숫자)
df['day'] = df.datetime.dt.day                        # 일(숫자)
df['hour'] = df.datetime.dt.hour                      # 시(숫자)
df['weekday'] = df.datetime.dt.weekday                # 요일(숫자)
df['dayofyear'] = df.datetime.dt.dayofyear            # 365일 중 얼마에 해당하는지
df['holiday'] = df['datetime'].apply(lambda x: 1 if x in kr_holidays else 0)  # 공휴일

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['rolling_mean'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['rolling_std'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).std())

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

#==============================================
# 날짜별로 모든 건물의 전력 소비량을 합산
df_total = df.groupby('datetime')['target'].mean()

# 시계열 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(df_total.index, df_total.values, color='blue', label='Total Power Consumption')

# 그래프 제목 및 레이블 설정
plt.title('Total Power Consumption Over Time Mean(All Buildings)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Total Power Consumption (kWh)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# 그래프 출력
plt.show()
#============================================
# 'weekday' 열이 이미 있는 상태에서 요일별로 데이터를 그룹화하여 평균 전력 사용량을 계산
df['weekday'] = df['datetime'].dt.weekday  # 0: 월요일 ~ 6: 일요일

# 요일별 평균 전력 사용량 계산
weekday_avg = df.groupby('weekday')['target'].mean()

# 요일별 전력 소비 히스토그램 그리기
plt.figure(figsize=(10, 6))
plt.bar(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], weekday_avg, color='skyblue')

# 그래프 제목 및 레이블 설정
plt.title('Average Power Consumption by Day of the Week', fontsize=16)
plt.xlabel('Day of the Week', fontsize=12)
plt.ylabel('Average Power Consumption (kWh)', fontsize=12)
plt.grid(axis='y')

# 그래프 출력
plt.show()
#============================================
# 'hour' 열이 이미 있는 상태에서 주간(6시~22시), 야간(22시~6시)으로 나누기
df['hour'] = df['datetime'].dt.hour

# 주간(9시 ~ 18시)과 야간(18시 ~ 9시) 데이터로 구분
df_daytime = df[(df['hour'] >= 4) & (df['hour'] < 17)]
df_nighttime = df[(df['hour'] >= 17) | (df['hour'] < 4)]

# 주간과 야간 시간대에서의 평균 전력 소비량 계산
daytime_avg = df_daytime.groupby('datetime')['target'].mean()
nighttime_avg = df_nighttime.groupby('datetime')['target'].mean()

# 서브플롯 생성 (2개의 서브플롯: 주간, 야간)
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)

# 주간 데이터 그래프 그리기
axes[0].plot(daytime_avg.index, daytime_avg.values, color='orange', label='Daytime (4 AM - 5 PM)')
axes[0].set_title('Daytime Power Consumption (4 AM - 5 PM)')
axes[0].set_ylabel('Average Power Consumption (kWh)')
axes[0].grid(True)

# 야간 데이터 그래프 그리기
axes[1].plot(nighttime_avg.index, nighttime_avg.values, color='blue', label='Nighttime (5 PM - 4 AM)')
axes[1].set_title('Nighttime Power Consumption (5 PM - 4 AM)')
axes[1].set_ylabel('Average Power Consumption (kWh)')
axes[1].grid(True)

# 공통 x축 레이블 설정
plt.xlabel('Date')
plt.tight_layout()
plt.show()
#============================================
fig, axes = plt.subplots(nrows=10, ncols=6, figsize=(20, 30))
axes = axes.flatten()  

# num 값 1부터 60까지 반복
for i in range(1, 61):
    # num이 i인 데이터를 필터링
    num_group = df[df['num'] == i][['datetime', 'target']]

    # 서브플롯에 그래프 그리기
    axes[i-1].plot(num_group['datetime'], num_group['target'], label=f'Building {i}')
    axes[i-1].set_title(f'Building {i}')
    axes[i-1].set_xlabel('Date Time')
    axes[i-1].set_ylabel('Power Usage')
    axes[i-1].grid(True)
    axes[i-1].legend()

# 서브플롯 간 레이아웃 조정
plt.tight_layout()

# 그래프 표시
plt.show()
#============================================
# 각 건물의 전력 소비량 평균 계산
building_avg_consumption = df.groupby('num')['target'].mean()

# 확률 밀도 함수(PDF) 그리기
plt.figure(figsize=(10, 6))
sns.kdeplot(building_avg_consumption, color='blue', shade=True)

# 그래프 제목 및 레이블 설정
plt.title('Probability Density Function of Average Power Consumption Across All Buildings', fontsize=16)
plt.xlabel('Average Power Consumption (kWh)', fontsize=12)
plt.ylabel('Density', fontsize=12)

# 그래프 출력
plt.grid(True)
plt.show()
#============================================
# target에 로그 변환 적용
df['log_target'] = np.log(df['target'] + 1)

# 각 건물의 전력 소비량 평균 계산 (로그 변환된 값)
building_avg_consumption_log = df.groupby('num')['log_target'].mean()

# 로그 변환된 확률 밀도 함수(PDF) 그리기
plt.figure(figsize=(10, 6))
sns.kdeplot(building_avg_consumption_log, color='blue', shade=True)

# 그래프 제목 및 레이블 설정
plt.title('Probability Density Function of Log-Transformed Average Power Consumption', fontsize=16)
plt.xlabel('Log(Average Power Consumption)', fontsize=12)
plt.ylabel('Density', fontsize=12)

# 그래프 출력
plt.grid(True)
plt.show()
#============================================
df.drop("datetime",axis=1,inplace=True)

scaler = StandardScaler()
scaled_columns = ['temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

df = pd.get_dummies(df, columns=['weekday'], drop_first=True)

train = []
valid = []
for num, group in df.groupby('num'):
    train.append(group.iloc[:len(group)-7*24])  
    valid.append(group.iloc[len(group)-7*24:]) 

train_df = pd.concat(train)
train_x = train_df.drop(["target", "diff_target"], axis=1)
train_y = train_df["diff_target"]

valid_df = pd.concat(valid)
valid_x = valid_df.drop(["target", "diff_target"], axis=1)
valid_y = valid_df["diff_target"]

# catboost
cat_model = CatBoostRegressor(random_state=42, verbose=0)  # verbose=0: 학습 로그 출력하지 않음
cat_model.fit(train_x, train_y)

cat_pred_diff = cat_model.predict(valid_x)

#=======================================================
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
xgb_pred_diff = np.exp(xgb_pred_log) - 1

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
lgb_pred_diff = lgb_best_model.predict(valid_x)

# 차분된 값을 원래 스케일로 복원
valid_df['target_shifted'] = valid_df.groupby('num')['target'].shift(1).fillna(method='bfill')
xgb_pred_final = valid_df['target_shifted'] + xgb_pred_diff
lgb_pred_final = valid_df['target_shifted'] + lgb_pred_diff
cat_pred_final = valid_df['target_shifted'] + cat_pred_diff

# 앙상블 가중치 적용 (CatBoost에 더 큰 가중치 부여)
ensemble_pred_weighted = (0.3 * xgb_pred_final + 0.2 * lgb_pred_final + 0.5 * cat_pred_final)

# 성능 평가
ensemble_mse_weighted = mean_squared_error(valid_df['target'], ensemble_pred_weighted)
print("Weighted Ensemble RMSE with Differencing:", np.sqrt(ensemble_mse_weighted))