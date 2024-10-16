import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_raw = pd.read_csv("data_week2.csv", encoding="CP949")
df = df_raw.copy()
df["date_time"].head(10)
plt.clf()
# Setting up plot style
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.tight_layout()
# Plotting the distribution of key variables
plt.figure(figsize=(14, 10))

# Plotting the histogram for temperature (기온(°C))
plt.subplot(2, 2, 1)
sns.histplot(df["기온(°C)"], bins=20, kde=True)
plt.title("Distribution of Temperature (기온(°C))")
plt.show()
# Plotting the histogram for wind speed (풍속(m/s))
plt.subplot(2, 2, 2)
sns.histplot(df["풍속(m/s)"], bins=20, kde=True)
plt.title("Distribution of Wind Speed (풍속(m/s))")
plt.show()
# Plotting the histogram for humidity (습도(%))
plt.subplot(2, 2, 3)
sns.histplot(df["습도(%)"], bins=20, kde=True)
plt.title("Distribution of Humidity (습도(%))")
plt.show()


# Plotting the histogram for 강수량(mm)
plt.subplot(2, 2, 1)
sns.histplot(df["강수량(mm)"], bins=20, kde=True)
plt.title("Distribution of Rainfall (강수량(mm))")
plt.show()



# Plotting the histogram for 일조(hr)
plt.subplot(2, 2, 2)
sns.histplot(df["일조(hr)"], bins=20, kde=True)
plt.title("Distribution of Sunshine Hours (일조(hr))")
plt.show()
# Plotting the histogram for 비전기냉방설비운영
plt.subplot(2, 2, 3)
sns.histplot(df["비전기냉방설비운영"], bins=2, kde=False)
plt.title("비전기냉방설비 보유 여부")
plt.show()
# Plotting the histogram for 태양광보유
plt.subplot(2, 2, 4)
sns.histplot(df["태양광보유"], bins=2, kde=False)
plt.title("태양광 보유 여부")
plt.show()



# Plotting the histogram for power consumption (전력사용량(kWh))
plt.subplot(2, 2, 4)
sns.histplot(df["전력사용량(kWh)"], bins=20, kde=True)
plt.title("Distribution of Power Consumption (전력사용량(kWh))")
plt.show()


df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
numerical_data_renamed = df[['target', 'temp', 'wind', 'humid', 'rain', 'sunny']]
correlation_matrix_renamed = numerical_data_renamed.corr()

# Visualizing the correlation matrix with the new column names
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_renamed, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix with Renamed Columns')
plt.show()


df_mean = df.groupby("num").agg(mean = ("target","mean"))
plt.bar(df_mean.index, df_mean["mean"])
plt.xlabel('Num')
plt.ylabel('Mean Target')
plt.title('Mean Target by Num Group')
plt.show()