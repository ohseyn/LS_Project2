# =====================주거용 그래프================================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows 환경에서는 Malgun Gothic 사용)
plt.rc('font', family='Malgun Gothic')  # 또는 다른 시스템 폰트 사용 가능

# Load the CSV file with the correct encoding
data = pd.read_csv('./data/house_sample.csv', encoding='cp949')

# Rename columns for easier manipulation
data.columns = ['시간대', '6월_월요일', '6월_근무일', '6월_토요일', '6월_일요일', 
                '7월_월요일', '7월_근무일', '7월_토요일', '7월_일요일', 
                '8월_월요일', '8월_근무일', '8월_토요일', '8월_일요일']

# Remove the first two non-numeric rows (metadata)
data = data.iloc[2:].reset_index(drop=True)

# Clean '시간대' column and convert to integer
data['시간대'] = data['시간대'].str.replace('시', '').astype(int)

# Calculate the mean for 평일 (월요일, 근무일) and 주말 (토요일, 일요일) for each month
data['6월_평일'] = data[['6월_월요일', '6월_근무일']].astype(float).mean(axis=1)
data['6월_주말'] = data[['6월_토요일', '6월_일요일']].astype(float).mean(axis=1)

data['7월_평일'] = data[['7월_월요일', '7월_근무일']].astype(float).mean(axis=1)
data['7월_주말'] = data[['7월_토요일', '7월_일요일']].astype(float).mean(axis=1)

data['8월_평일'] = data[['8월_월요일', '8월_근무일']].astype(float).mean(axis=1)
data['8월_주말'] = data[['8월_토요일', '8월_일요일']].astype(float).mean(axis=1)

# Plotting the average power consumption coefficients for 평일 and 주말 by hour for each month
plt.figure(figsize=(10, 6))

# 6월: 파란색, 7월: 초록색, 8월: 빨간색
plt.plot(data['시간대'], data['6월_평일'], label='6월 평일', linestyle='-', color='blue')
plt.plot(data['시간대'], data['6월_주말'], label='6월 주말', linestyle='--', color='blue')

plt.plot(data['시간대'], data['7월_평일'], label='7월 평일', linestyle='-', color='green')
plt.plot(data['시간대'], data['7월_주말'], label='7월 주말', linestyle='--', color='green')

plt.plot(data['시간대'], data['8월_평일'], label='8월 평일', linestyle='-', color='red')
plt.plot(data['시간대'], data['8월_주말'], label='8월 주말', linestyle='--', color='red')

plt.title('2020년 6, 7, 8월 시간대별 평일 및 주말 전력소비계수 평균')
plt.xlabel('시간대 (시)')
plt.ylabel('전력소비계수')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 25))
plt.show()

# ======================산업용 그래프 8월=============================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math

# 한글 폰트 설정 (Windows 환경에서는 Malgun Gothic 사용)
plt.rc('font', family='Malgun Gothic')

# Load the dataset
data = pd.read_csv('./data/industry_sample.csv', encoding='cp949')

# Set the first row as the header and reset the dataframe
data.columns = data.iloc[0]
data = data.drop(0).reset_index(drop=True)

# Clean '시간별' column by removing '시' and converting it to integer
data['시간별'] = data['시간별'].str.replace('시', '').astype(int)

# Extract industry names (excluding '월별' and '시간별')
industry_names = data.columns[2:]

# Convert all industry columns to numeric values
data[industry_names] = data[industry_names].apply(pd.to_numeric, errors='coerce')

# Filter data for only '8월'
august_data = data[data['월별'] == '8월']

# Set the number of rows and columns for the subplots grid
n_rows = 5
n_cols = 5

# Calculate the number of figures needed
num_figures = math.ceil(len(industry_names) / (n_rows * n_cols))

# Create multiple figures, each showing a subset of industries
for fig_num in range(num_figures):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    axes = axes.flatten()
    
    # Select a subset of industries for this figure
    start_idx = fig_num * n_rows * n_cols
    end_idx = min(start_idx + n_rows * n_cols, len(industry_names))
    industries_subset = industry_names[start_idx:end_idx]
    
    # Plot each industry in its own subplot
    for idx, industry in enumerate(industries_subset):
        axes[idx].plot(august_data['시간별'], august_data[industry], label=industry)
        axes[idx].set_title(industry, fontsize=10)  # Set the title as the industry name
        axes[idx].set_xlabel('시간대 (시)', fontsize=8)
        axes[idx].set_ylabel('전력소비계수', fontsize=8)
        axes[idx].grid(True)
        axes[idx].set_xticks(range(1, 25))  # Ensure x-axis ticks are 1-24
    
    # Hide any unused subplots in the last figure
    for empty_plot_idx in range(idx + 1, len(axes)):
        fig.delaxes(axes[empty_plot_idx])
    
    # Adjust layout and show the figure
    plt.suptitle(f'2020년 8월 산업별 시간대별 전력소비계수 (산업 {start_idx + 1} ~ {end_idx})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()