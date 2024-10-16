import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import holidays

# 데이터 로드
df_raw = pd.read_csv("data_week2.csv", encoding="CP949")
df = df_raw.copy()

# 한국 공휴일 설정
kr_holidays = holidays.KR()


df["target"].index
plt.plot(df["target"].index,df["target"])








# 컬럼명 수정
df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])

# 새로운 특성 추가
df['month'] = df.datetime.dt.month                    # 월(숫자)
df['day'] = df.datetime.dt.day                        # 일(숫자)
df['hour'] = df.datetime.dt.hour                      # 시(숫자)
df['weekday'] = df.datetime.dt.weekday                # 요일(숫자)
df['dayofyear'] = df.datetime.dt.dayofyear            # 365일 중 몇 번째 날인지
df['holiday'] = df['datetime'].apply(lambda x: 1 if x in kr_holidays else 0)  # 공휴일

# 시간 특성 생성 (주기적 특성)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Rolling Mean 및 Rolling Std 추가
df['rolling_mean'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['rolling_std'] = df.groupby('num')['target'].transform(lambda x: x.rolling(window=3, min_periods=1).std())

# 스케일링 (StandardScaler 사용)
scaler = StandardScaler()
scaled_columns = ['temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

# datetime 컬럼 제거
df.drop("datetime", axis=1, inplace=True)

# 피처 중요도 시각화 함수
def calculate_lgbm_for_num(df, num):
    num_df = df[df["num"] == num]
    X = num_df.drop(columns=["target"])
    y = num_df["target"]
    train_size = len(X) - 168
    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]
    
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # 피처 중요도 계산
    lgbm_importance = model.feature_importances_
    return lgbm_importance

# 서브플롯 설정
fig, axes = plt.subplots(12, 5, figsize=(20, 40))  # 12행 5열로 60개 플롯
axes = axes.flatten()

# 각 건물의 피처 중요도를 서브플롯에 그리기
for num in range(1, 61):  # 건물 번호 1부터 60까지
    importance = calculate_lgbm_for_num(df, num)
    features = df.drop(columns=["target"]).columns
    
    axes[num-1].barh(features, importance)
    axes[num-1].set_title(f'Building {num}')

# 그래프 레이아웃 조정
plt.tight_layout()
plt.show()
