import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import holidays

# 한국 공휴일 정보
kr_holidays = holidays.KR()

# 데이터 로드
df_raw = pd.read_csv("data_week2.csv", encoding="CP949")

# 데이터 복사
df = df_raw.copy()

# 컬럼명 재설정
df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']

# 날짜/시간형 데이터로 변환
df["datetime"] = pd.to_datetime(df["datetime"])

# 추가적인 날짜 정보 생성
df['month'] = df.datetime.dt.month  # 월
df['day'] = df.datetime.dt.day  # 일
df['hour'] = df.datetime.dt.hour  # 시
df['weekday'] = df.datetime.dt.weekday  # 요일
df['holiday'] = df['datetime'].apply(lambda x: 1 if x in kr_holidays else 0)  # 공휴일 여부
df['week'] = (df['datetime'] - df['datetime'].min()).dt.days // 7 + 1  # 주차

# 'datetime'을 인덱스로 설정
df.set_index('datetime', inplace=True)

# 필요한 feature와 target 설정
features = ['temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar', 'month', 'day', 'hour', 'weekday', 'holiday']
target = ['target']

# 데이터 스케일링 (MinMaxScaler 사용)
scaler = MinMaxScaler()
df[features + target] = scaler.fit_transform(df[features + target])

# LSTM에 사용할 데이터를 시계열 형태로 만들기
def create_sequences(data, target_col, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data.iloc[i:i + seq_length][features].values)  # 시퀀스 추가
        targets.append(data.iloc[i + seq_length][target_col])  # 단일 값으로 타겟 추가
    return np.array(sequences), np.array(targets)

# 시퀀스 길이 설정 (여기서는 24시간을 하나의 시퀀스로 설정)
seq_length = 24
X, y = create_sequences(df, 'target', seq_length)

# 데이터 분할 (train-test split)
train = []
valid = []
for num, group in df.groupby('num'):
    train.append(group.iloc[:len(group)-7*24])  
    valid.append(group.iloc[len(group)-7*24:]) 
train_df = pd.concat(train)
X_train = train_df.drop("target",axis=1)
y_train = train_df["target"] 

test_df = pd.concat(valid)
X_test = test_df.drop("target",axis=1)
y_test = test_df["target"] 

# 시퀀스 길이 설정 (여기서는 24시간을 하나의 시퀀스로 설정)
seq_length = 24

# train과 test 데이터를 시퀀스로 변환
X_train_seq, y_train_seq = create_sequences(train_df, 'target', seq_length)
X_test_seq, y_test_seq = create_sequences(test_df, 'target', seq_length)

# 데이터를 numpy 배열로 변환 후 텐서로 변환
X_train = torch.tensor(X_train_seq, dtype=torch.float32)
X_test = torch.tensor(X_test_seq, dtype=torch.float32)
y_train = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1)  # 2D 텐서로 변환
y_test = torch.tensor(y_test_seq, dtype=torch.float32).view(-1, 1)  # 2D 텐서로 변환


# GRU 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)  # GRU에 입력 데이터 전달
        out = self.fc(out[:, -1, :])  # 마지막 시퀀스의 출력만 사용
        return out

# 하이퍼파라미터 설정
input_size = len(features)  # feature 개수
hidden_size = 100  # hidden state 크기
num_layers = 3  # GRU 레이어 수
output_size = 1  # 예측할 타겟 크기 (단일 값 예측)
learning_rate = 0.001
num_epochs = 100

# 모델, 손실함수, 옵티마이저 정의
model = GRUModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 과정
model.train()
for epoch in range(num_epochs):
    outputs = model(X_train)  # X_train의 shape이 (batch_size, seq_length, num_features)인지 확인
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 성능지표-----------------------------------
# 평가
# SMAPE 함수 정의
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# 모델 평가
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.detach().numpy()
    y_test_np = y_test.numpy()

    # MSE 계산
    mse = mean_squared_error(y_test_np, y_pred)

    # RMSE 계산
    rmse = np.sqrt(mse)

    # SMAPE 계산
    smape_value = smape(y_test_np, y_pred)

    # 결과 출력
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'SMAPE: {smape_value:.4f}%')

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_test_np, label='Actual')
plt.plot(y_pred, label='Bi-LSTM Prediction', color='red')
plt.legend()
plt.show()