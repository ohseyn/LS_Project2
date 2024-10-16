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
import matplotlib.pyplot as plt

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
# 일주일 단위로 주차(week number) 생성
df['week'] = (df['datetime'] - df['datetime'].min()).dt.days // 7 + 1
df.drop("datetime",axis=1,inplace=True)

df.info()
df.head()

# 주차별 변동성 분석 (표준편차)
weekly_std = df.groupby('week').std()
plt.figure(figsize=(10, 6))
plt.plot(weekly_std.index, weekly_std['target'], marker='o', color='red', label='Target Std Dev')
plt.title('Weekly Standard Deviation of Target (Power Consumption)')
plt.xlabel('Week')
plt.ylabel('Standard Deviation of Power Consumption')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# LightGBM 또는 XGBoost를 사용해 주별로 Feature Importance 분석
model = LGBMRegressor()
X = df.drop(columns=['target', 'day'])  # 독립 변수
y = df['target']  # 종속 변수 (전력 소비량)

model.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

# Feature Importance 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance (LGBM)')
plt.tight_layout()
plt.show()

# 1~13주차 데이터를 필터링
df_filtered = df[df['week'].between(1, 13)]

# 주차별 전력 소비량의 평균 계산
weekly_avg_target = df_filtered.groupby('week')['target'].mean()

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(weekly_avg_target.index, weekly_avg_target.values, marker='o', linestyle='-', color='blue')
plt.title('Average Power Consumption by Week (1~13)')
plt.xlabel('Week')
plt.ylabel('Average Power Consumption (kWh)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 건물별 주차 전력량
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn 팔레트에서 건물 수에 맞는 색상 선택
colors = sns.color_palette("husl", df['num'].nunique())

# 1~13주차 데이터를 필터링
df_filtered = df[df['week'].between(1, 13)]

# 주차별 전력 사용량을 시각화
plt.figure(figsize=(15, 20))

for week in range(1, 14):
    plt.subplot(7, 2, week)  # 7행 2열의 서브플롯 구성
    week_data = df_filtered[df_filtered['week'] == week]

    # 건물별로 선 색상을 다르게 설정하여 시각화
    for i, building in enumerate(week_data['num'].unique()):
        building_data = week_data[week_data['num'] == building]
        plt.plot(building_data['hour'], building_data['target'], marker='o', linestyle='-', color=colors[i], label=f'Building {building}')
    
    plt.title(f'Week {week} Power Consumption')
    plt.xlabel('Hour of Day')
    plt.ylabel('Power Consumption (kWh)')
    plt.grid(True)

plt.tight_layout()
plt.show()

