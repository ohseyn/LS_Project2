import pandas as pd
import numpy as np
df = pd.read_csv('data_week2.csv', encoding = 'cp949')

## 변수들을 영문명으로 변경
cols = ['num', 'date_time', 'power', 'temp', 'wind','hum' ,'prec', 'sun', 'non_elec', 'solar']
df.columns = cols

df.info()
df.head()

# 시간 관련 변수들 생성
df['date_time'] = pd.to_datetime(df['date_time'])

# 연도, 월, 일, 시 등을 새로운 컬럼으로 분리
# df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour

## 상관관계 분석
# non_elec, solar 범주형
import seaborn as sns
import matplotlib.pyplot as plt

# 상관관계 분석을 위한 열 선택
columns_of_interest = ['power', 'temp', 'wind','hum' ,'prec', 'sun']

# 상관행렬 계산
corr_matrix = df[columns_of_interest].corr()

# 상관행렬 시각화
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Correlation Matrix for Electricity Usage and Weather Variables')
plt.show()

import matplotlib.pyplot as plt

# Plotting each building's electricity usage on separate plots
num_buildings = df['num'].nunique()

# Set up the plotting environment
fig, axes = plt.subplots(nrows=15, ncols=4, figsize=(20, 30))  # Adjusting for 60 buildings
axes = axes.flatten()  # Flattening the array of axes for easier iteration

# Plot each building's data on a separate subplot
for i, building in enumerate(df['num'].unique()):
    building_data = df[df['num'] == building]
    axes[i].plot(building_data['date_time'], building_data['power'])
    axes[i].set_title(f'Building {building}')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Electricity Usage (kWh)')
    axes[i].grid(True)

# Adjust the layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

# 시간별 평균 전력량
# Set up the plotting environment for 60 buildings (15 rows x 4 columns)
fig, axes = plt.subplots(nrows=15, ncols=4, figsize=(20, 40))
axes = axes.flatten()  # Flatten the axes array for easier iteration

# Plot each building's data on a separate subplot
for i, building in enumerate(df['num'].unique()):
    # Filter data for the current building
    building_data = df[df['num'] == building]

    # Extracting 3 months of data - June, July, and August
    building_3_months = building_data[(building_data['date_time'] >= '2020-06-01') & (building_data['date_time'] < '2020-09-01')]

    # Convert date_time to only hours to calculate the hourly averages
    building_3_months['hour'] = building_3_months['date_time'].dt.hour

    # Group by hour to calculate the average electricity usage for each hour
    hourly_avg_electricity_usage = building_3_months.groupby('hour')['power'].mean()

    # Plot hourly average electricity usage
    axes[i].plot(hourly_avg_electricity_usage.index, hourly_avg_electricity_usage.values, marker='o')
    axes[i].set_title(f'Building {building}', fontsize=10)
    axes[i].set_xlabel('Hour of the Day')
    axes[i].set_ylabel('Avg Electricity (kWh)')
    axes[i].grid(True)

# Adjust the layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

# 시간, 일, 월별 전기 사용량 추세
# Filter the data for one building (e.g., Building 1)
building_data = df[df['num'] == 5]

# 1. Hourly Trend
building_data['hour'] = building_data['date_time'].dt.hour
hourly_trend = building_data.groupby('hour')['power'].mean()

# 2. Daily Trend
building_data['day'] = building_data['date_time'].dt.date
daily_trend = building_data.groupby('day')['power'].mean()

# 3. Monthly Trend
building_data['month'] = building_data['date_time'].dt.to_period('M')
monthly_trend = building_data.groupby('month')['power'].mean()

# Plotting the trends
plt.figure(figsize=(15, 12))

# Hourly Trend
plt.subplot(3, 1, 1)
plt.plot(hourly_trend.index, hourly_trend.values, marker='o')
plt.title('Hourly Electricity Usage Trend', fontsize=14)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Average Electricity Usage (kWh)', fontsize=12)
plt.grid(True)

# Daily Trend
plt.subplot(3, 1, 2)
plt.plot(daily_trend.index, daily_trend.values, marker='o')
plt.title('Daily Electricity Usage Trend', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Electricity Usage (kWh)', fontsize=12)
plt.grid(True)

# Monthly Trend
plt.subplot(3, 1, 3)
plt.plot(monthly_trend.index.astype(str), monthly_trend.values, marker='o')
plt.title('Monthly Electricity Usage Trend', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Electricity Usage (kWh)', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()

# 차분 시도
# 1. Resample the data to weekly frequency and sum the electricity usage for each week
weekly_data = df.resample('W', on='date_time')['power'].sum()

# 2. Calculate the difference between each week's electricity usage (1-week differencing)
weekly_data_diff = weekly_data.diff()

# Display the first few rows of the weekly data and the difference
print(weekly_data.head())  # Weekly electricity usage
print(weekly_data_diff.head())  # Weekly difference (차분)

#Plotting the weekly difference using a bar chart
plt.figure(figsize=(10, 6))

# Bar plot for weekly differencing
plt.bar(weekly_data_diff.index, weekly_data_diff.values, color='orange')
plt.title('Weekly Difference in Electricity Usage')
plt.xlabel('Week')
plt.ylabel('Difference in Electricity Usage (kWh)')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# -------------------------------
num_1 = df[df['num'] == 1]
