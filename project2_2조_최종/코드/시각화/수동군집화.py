import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./data/data_week2.csv', encoding='cp949')

# ====================Feature Importance==================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('./data/data_week2.csv', encoding='cp949')

# date_time을 datetime 형태로 변환
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H')

# 필요 없는 칼럼 제거 및 독립변수와 종속변수 정의
# 종속변수: 전력 사용량(kWh), 독립변수: 나머지 변수들 (건물번호 포함)
X = data[['num', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유']]  # 독립변수
y = data['전력사용량(kWh)']  # 종속변수

# 결측치 처리 (간단히 평균값으로 대체)
X.fillna(X.mean(), inplace=True)

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 훈련
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Feature Importance 계산
importances = model.feature_importances_

# Feature Importance 시각화
feature_names = X.columns
indices = np.argsort(importances)[::-1]

# 막대마다 다른 색을 적용
colors = plt.cm.get_cmap('tab10', len(feature_names))

plt.figure(figsize=(10, 6))
plt.title('Feature Importance - Power Consumption')
plt.bar(range(X.shape[1]), importances[indices], color=colors(range(X.shape[1])), align='center')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.tight_layout()
plt.show()

# 각 독립변수의 중요도를 출력
for f in range(X.shape[1]):
    print(f"{f + 1}. Feature '{feature_names[indices[f]]}' Importance: {importances[indices[f]]:.3f}")



# ================전체 평균량 그래프======================
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data/data_week2.csv', encoding='cp949')


# date_time을 datetime 타입으로 변환
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H')

# 요일 정보를 추가 (0=월요일, 6=일요일)
data['weekday'] = data['date_time'].dt.dayofweek

# 주말(토요일=5, 일요일=6)을 제외한 데이터 필터링
weekday_data = data[data['weekday'] < 5]

# 시간대만 추출 (hour)
weekday_data['hour'] = weekday_data['date_time'].dt.hour

# 건물별로 시간대에 따른 평균 전력 사용량 계산
building_hourly_avg = weekday_data.groupby(['num', 'hour'])['전력사용량(kWh)'].mean().unstack()

# 선의 굵기 설정
line_width = 3  # 원하는 선 굵기를 여기에서 조절하세요

# 전체 그래프 크기 설정 (10x6개의 서브플롯 생성)
fig, axes = plt.subplots(10, 6, figsize=(20, 20), sharex=True, sharey=False)
fig.suptitle('Weekday Hourly Average Power Consumption by Building', fontsize=16)

# x축에 0시부터 24시까지 시각 설정
xticks = range(0, 25, 6)

# 각 건물별 그래프 그리기
for i, (building_num, ax) in enumerate(zip(building_hourly_avg.index, axes.flatten())):
    # 건물별 평균 전력 사용량 플롯 (주황색 선 그래프)
    ax.plot(building_hourly_avg.columns, building_hourly_avg.loc[building_num], color='blue', linestyle='-', linewidth=line_width)
    ax.set_title(f'Building {building_num}', fontsize=8)
    
    # x축을 6시간 단위로 0시부터 24시까지 표시
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{x}:00' for x in xticks], fontsize=8)  # 시간 레이블 추가
    
    # y축 범위를 건물별 전력 사용량의 최저점과 최고점에 맞춰 조정 (공백 최소화)
    min_val = building_hourly_avg.loc[building_num].min()
    max_val = building_hourly_avg.loc[building_num].max()
    ax.set_ylim([min_val - (max_val - min_val) * 0.05, max_val + (max_val - min_val) * 0.05])
    
    # 그리드 추가
    ax.grid(True)
    
    # y축 라벨 설정 (첫 번째 열만)
    if i % 6 == 0:
        ax.set_ylabel('Power (kWh)', fontsize=8)
    
    # x축 라벨 설정 (모든 행에서 시간을 표시)
    ax.set_xlabel('Hour of the Day', fontsize=8)

# 서브플롯 간 간격 조정
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ==============비슷한 그래프끼리 모아서 그리기====================================
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정 (맑은 고딕)
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕 폰트 경로 (Windows 경로)
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# date_time을 datetime 타입으로 변환
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H')

# 요일 정보를 추가 (0=월요일, 6=일요일)
data['weekday'] = data['date_time'].dt.dayofweek

# 주말(토요일=5, 일요일=6)을 제외한 데이터 필터링
weekday_data = data[data['weekday'] < 5]

# 시간대만 추출 (hour)
weekday_data['hour'] = weekday_data['date_time'].dt.hour

# 건물별로 시간대에 따른 평균 전력 사용량 계산
building_hourly_avg = weekday_data.groupby(['num', 'hour'])['전력사용량(kWh)'].mean().unstack()

# 건물 그룹 정의
group_1 = [2, 4, 10, 11, 12, 16, 22, 28, 29, 36, 38, 39, 41, 46, 47, 53, 58, 59]  # 모자 모양 (60 제거)
group_2 = [6, 8, 13, 14, 17, 18, 24, 25, 26, 27, 31, 33, 35, 43, 44, 48, 52, 54, 55, 56, 60]  # 가운데 파인 모자 (60 추가)
group_3 = [7, 23, 30, 32, 37, 40, 42]  # 팔을 올린 모자
group_4 = [20, 21, 49, 50, 51]  # 볼록볼록
group_5 = [1, 3, 5, 9, 15, 34, 45, 57]  # 특이한 그래프

# 그룹과 그룹 이름
groups = [group_1, group_2, group_3, group_4, group_5]
group_names = ['그룹 1', '그룹 2', '그룹 3', '그룹 4', '그룹 5']

# 선의 굵기 설정
line_width = 2  # 원하는 선 굵기를 여기에서 조절하세요

# 그룹별로 그래프 생성
for group_idx, group in enumerate(groups):
    # 서브플롯의 행과 열 수를 동적으로 설정
    n_buildings = len(group)
    n_cols = 5
    n_rows = (n_buildings + n_cols - 1) // n_cols  # 행의 개수는 건물 수에 맞춰 계산

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))  # 동적으로 서브플롯 크기 설정
    fig.suptitle(group_names[group_idx], fontsize=16)

    # x축에 0시부터 24시까지 시각 설정 (6시간 단위)
    xticks = [0, 6, 12, 18, 24]

    # 각 건물별 그래프 그리기
    for i, building_num in enumerate(group):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]  # n_rows가 1일 때 처리를 위해 조건 추가

        if building_num in building_hourly_avg.index:
            ax.plot(building_hourly_avg.columns, building_hourly_avg.loc[building_num], color='blue', linestyle='-', linewidth=line_width)
            ax.set_title(f'Building {building_num}', fontsize=8)
            ax.set_xticks(xticks)
            ax.set_xticklabels(['0', '6', '12', '18', '24'], fontsize=8)
            ax.grid(True)

            # y축 범위 조정 (각 건물 그래프 간 차이를 반영)
            min_val = building_hourly_avg.loc[building_num].min()
            max_val = building_hourly_avg.loc[building_num].max()
            ax.set_ylim([min_val - (max_val - min_val) * 0.1, max_val + (max_val - min_val) * 0.1])

    # 빈 서브플롯 처리 (그래프가 없는 영역을 숨기기 위함)
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ===================특징 도출하기(그룹별 전력사용량 평균 및 기후)==================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드
data = pd.read_csv('./data/data_week2.csv', encoding='cp949')

# date_time을 datetime 형태로 변환
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H')

# 평일 데이터만 선택 (평일: 월=0, 화=1, 수=2, 목=3, 금=4)
data['weekday'] = data['date_time'].dt.weekday
weekday_data = data[data['weekday'] < 5]

# 그룹 리스트 정의 (그룹 5번은 제외)
group_1 = [2, 4, 10, 11, 12, 16, 22, 28, 29, 36, 38, 39, 41, 46, 47, 53, 58, 59]
group_2 = [6, 8, 13, 14, 17, 18, 24, 25, 26, 27, 31, 33, 35, 43, 44, 48, 52, 54, 55, 56, 60]
group_3 = [7, 23, 30, 32, 37, 40, 42]
group_4 = [20, 21, 49, 50, 51]

# 그룹 할당 함수
def assign_group(num):
    if num in group_1:
        return 1
    elif num in group_2:
        return 2
    elif num in group_3:
        return 3
    elif num in group_4:
        return 4
    return np.nan  # 그룹 5는 제외

# 그룹 번호 할당
weekday_data['group'] = weekday_data['num'].apply(assign_group)

# 시각화 함수 정의
def plot_comparison(column_name, y_label):
    groups = [1, 2, 3, 4]
    colors = ['blue', 'orange', 'green', 'red']
    
    plt.figure(figsize=(10, 6))
    
    for i, group in enumerate(groups):
        group_data = weekday_data[weekday_data['group'] == group]
        group_avg = group_data.groupby(group_data['date_time'].dt.date)[column_name].mean()
        
        plt.plot(group_avg.index, group_avg.values, label=f'Group {group}', color=colors[i])
    
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.title(f'Weekday Average {y_label} for Groups 1 to 4')
    plt.xticks(rotation=45)
    plt.legend()  # 범례 추가
    plt.tight_layout()
    plt.show()

# 전력사용량, 기온, 풍속, 습도, 강수량, 일조에 대해 각각의 그래프 그리기
columns_to_plot = {
    '전력사용량(kWh)': 'Power Consumption (kWh)',
    '기온(°C)': 'Temperature (°C)',
    '풍속(m/s)': 'Wind Speed (m/s)',
    '습도(%)': 'Humidity (%)',
    '강수량(mm)': 'Precipitation (mm)',
    '일조(hr)': 'Sunshine Hours (hr)'
}

for col, y_label in columns_to_plot.items():
    plot_comparison(col, y_label)


# ===================시설 비율 찾기=====================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드
data = pd.read_csv('./data/data_week2.csv', encoding='cp949')

# 그룹 리스트 정의 (그룹 5번은 제외)
group_1 = [2, 4, 10, 11, 12, 16, 22, 28, 29, 36, 38, 39, 41, 46, 47, 53, 58, 59]
group_2 = [6, 8, 13, 14, 17, 18, 24, 25, 26, 27, 31, 33, 35, 43, 44, 48, 52, 54, 55, 56, 60]
group_3 = [7, 23, 30, 32, 37, 40, 42]
group_4 = [20, 21, 49, 50, 51]

# 그룹 할당 함수
def assign_group(num):
    if num in group_1:
        return 1
    elif num in group_2:
        return 2
    elif num in group_3:
        return 3
    elif num in group_4:
        return 4
    return np.nan  # 그룹 5는 제외

# 그룹 번호 할당
data['group'] = data['num'].apply(assign_group)

# num에 해당하는 첫번째 열만 고려하여 중복 제거
first_entry_data = data.drop_duplicates(subset='num')

# 그룹별 비전기냉방설비운영 및 태양광보유 비율 계산 (분자/분모 형태)
group_stats_count = first_entry_data.groupby('group')[['비전기냉방설비운영', '태양광보유']].sum()
group_stats_total = first_entry_data.groupby('group')[['비전기냉방설비운영', '태양광보유']].count()

# 결과 출력 (분자/분모 형태와 소수점 형태)
print("그룹별 비전기냉방설비운영 및 태양광보유 (분자/분모 및 소숫점 비율):")
for group in group_stats_count.index:
    non_electric_op = f"{int(group_stats_count.loc[group, '비전기냉방설비운영'])}/{int(group_stats_total.loc[group, '비전기냉방설비운영'])}"
    solar_own = f"{int(group_stats_count.loc[group, '태양광보유'])}/{int(group_stats_total.loc[group, '태양광보유'])}"
    non_electric_op_ratio = group_stats_ratio.loc[group, '비전기냉방설비운영']
    solar_own_ratio = group_stats_ratio.loc[group, '태양광보유']
    print(f"그룹 {group}: 비전기냉방설비운영: {non_electric_op} ({non_electric_op_ratio:.2f}), 태양광보유: {solar_own} ({solar_own_ratio:.2f})")

# 비율 계산 (막대 그래프용)
group_stats_ratio = group_stats_count / group_stats_total

# 막대 그래프 그리기
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# 비전기냉방설비운영 비율 그래프
axs[0].bar(group_stats_ratio.index, group_stats_ratio['비전기냉방설비운영'], color=['blue', 'orange', 'green', 'red'])
axs[0].set_xlabel('Group')
axs[0].set_ylabel('Non-electric Cooling System Operation Ratio')
axs[0].set_title('Non-electric Cooling System Operation by Group')
axs[0].set_xticks(group_stats_ratio.index)  # x축을 정수로만 설정

# 태양광보유 비율 그래프
axs[1].bar(group_stats_ratio.index, group_stats_ratio['태양광보유'], color=['blue', 'orange', 'green', 'red'])
axs[1].set_xlabel('Group')
axs[1].set_ylabel('Solar Panel Ownership Ratio')
axs[1].set_title('Solar Panel Ownership by Group')
axs[1].set_xticks(group_stats_ratio.index)  # x축을 정수로만 설정

plt.tight_layout()
plt.show()

# =================전력사용량 박스플롯=====================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('./data/data_week2.csv', encoding='cp949')

# date_time을 datetime 형태로 변환
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H')

# 그래프 크기 설정
plt.figure(figsize=(12, 8))

# Seaborn 박스플롯 생성 (각 박스를 서로 다른 색으로 구분)
sns.boxplot(x='num', y='전력사용량(kWh)', data=data, fliersize=3)

# 그래프 제목 및 라벨 설정 (글자 크기 작게 설정)
plt.title('Power Consumption by Building Number (num)', fontsize=12)
plt.xlabel('Building Number (num)', fontsize=10)
plt.ylabel('Power Consumption (kWh)', fontsize=10)

# x축의 글자 크기 설정
plt.xticks(rotation=0, fontsize=5)
plt.yticks(fontsize=8)

# 그래프 표시
plt.tight_layout()
plt.show()

# ====================================================
