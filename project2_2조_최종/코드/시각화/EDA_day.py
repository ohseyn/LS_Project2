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

df.info()
df.head()

# 각 시간대별 건물의 평균 전력 소비량 계산
avg_power_hour = df.groupby('hour')['target'].mean()

plt.figure(figsize=(10, 6))
plt.plot(avg_power_hour.index, avg_power_hour.values, marker='o', linestyle='-', color='b')

plt.title('Average Power Consumption by Hour (All Buildings)')
plt.xlabel('Hour of Day')
plt.ylabel('Average Power Consumption (kWh)')
plt.grid(True)
plt.xticks(range(0, 24))  # X축을 0~23시간으로 설정
plt.tight_layout()
plt.show()

# 전력 소비량이 높은 시간대 (9시 ~ 17시)와 낮은 시간대 (0시 ~ 6시) 정의
high_consumption_hours = df[(df['hour'] >= 9) & (df['hour'] <= 17)]
low_consumption_hours = df[(df['hour'] >= 0) & (df['hour'] <= 6)]

# 높은 시간대와 낮은 시간대의 기후 변수 평균 계산
high_climate_avg = high_consumption_hours[['temp', 'wind', 'humid', 'rain', 'sunny']].mean()
low_climate_avg = low_consumption_hours[['temp', 'wind', 'humid', 'rain', 'sunny']].mean()

# 결과 출력
print("High Power Consumption Hours (9AM - 5PM):")
print(high_climate_avg)
print("\nLow Power Consumption Hours (12AM - 6AM):")
print(low_climate_avg)

# 기후 변수 비교를 위한 데이터 준비
climate_comparison = pd.DataFrame({
    'High Consumption': high_climate_avg,
    'Low Consumption': low_climate_avg
})

# 바 차트로 기후 변수 비교
climate_comparison.plot(kind='bar', figsize=(8, 6))
plt.title('Climate Variables Comparison (High vs Low Power Consumption Hours)')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()
