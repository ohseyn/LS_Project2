import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score,recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from A_catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import holidays
#!pip install holidays

kr_holidays = holidays.KR()

df_raw = pd.read_csv("data_week2.csv",encoding="CP949")

df = df_raw.copy()

df.columns

df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df['month'] = df.datetime.dt.month                    # 월(숫자)
df['day'] = df.datetime.dt.day                        # 일(숫자)
df['hour'] = df.datetime.dt.hour                      # 시(숫자)
df['weekday'] = df.datetime.dt.weekday                # 요일(숫자)
df['holiday'] = df['datetime'].apply(lambda x: 1 if x in kr_holidays else 0)  # 공휴일


# 공휴일과 공휴일이 아닌 날의 전력 소비량 비교
holiday_data = df[df['holiday'] == 1]
non_holiday_data = df[df['holiday'] == 0]

# 공휴일과 공휴일이 아닌 날의 전력 소비량 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(holiday_data['target'], bins=30, alpha=0.5, label='Holiday', color='blue')
plt.hist(non_holiday_data['target'], bins=30, alpha=0.5, label='Non-Holiday', color='orange')

plt.title('Power Consumption on Holidays vs Non-Holidays (All Buildings)')
plt.xlabel('Power Consumption (kWh)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()


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

# 31 이상치
# 6월 데이터 필터링
num31_6m = df[(df['num'] == 31) & (df['month'] == 6)]

# 전력 사용량이 최대인 날
max_power_day = num31_6m[num31_6m['target'] == num31_6m['target'].max()]

# 전력 사용량이 최소인 날
min_power_day = num31_6m[num31_6m['target'] == num31_6m['potargetwer'].min()]

# 결과 출력
print("\n전력 사용량이 가장 낮은 날:")
print(min_power_day[['date_time', 'power']]) # 0611

# 33 이상치
# 6월 데이터 필터링
num33_6m = df[(df['num'] == 33) & (df['month'] == 6)]

# 전력 사용량이 최대인 날
max_power_day = num33_6m[num33_6m['power'] == num33_6m['power'].max()]

# 전력 사용량이 최소인 날
min_power_day = num33_6m[num33_6m['power'] == num33_6m['power'].min()]

# 결과 출력
print("\n전력 사용량이 가장 낮은 날:")
print(min_power_day[['date_time', 'power']]) # 0611

# 6월달 전체 데이터와 비교해 해당 값이 이상치인지 확인
building_31_june = df[(df['num'] == 31) & (df['month'] == 6)]
building_31_june.describe()  # 전력 사용량 통계 정보 확인

