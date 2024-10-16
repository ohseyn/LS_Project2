import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

# 데이터 로드 및 전처리
df_raw = pd.read_csv("data_week2.csv", encoding="CP949")
df = df_raw.copy()

# 컬럼명 재설정 및 날짜 변환
df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index('datetime', inplace=True)

# SMAPE 함수 정의
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# 시퀀스를 만드는 함수 정의
def create_sequences(data, target, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data.iloc[i:i + seq_length].values)
        targets.append(target.iloc[i + seq_length])
    return np.array(sequences), np.array(targets)

# 성능 저장할 리스트 초기화
rmse_list = []
smape_list = []

# 서브플롯 설정 (10행 6열의 서브플롯)
fig, axs = plt.subplots(10, 6, figsize=(30, 20))
fig.suptitle('Bi-LSTM Model Predictions for All Buildings', fontsize=20)

# 그룹별로 Bi-LSTM 모델 학습 및 예측
for idx, (num, group) in enumerate(df.groupby('num')):
    print(f"Processing Building: {num}")
    
    # 데이터를 훈련과 검증으로 나눔 (검증 데이터는 마지막 7일의 데이터)
    train = group.iloc[:len(group)-7*24]
    valid = group.iloc[len(group)-7*24:]
    
    # 필요한 feature 설정
    features = ['temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
    
    # 시퀀스 길이 설정 (여기서는 24시간을 하나의 시퀀스로 설정)
    seq_length = 24

    # train과 test 데이터를 시퀀스로 변환
    X_train_seq, y_train_seq = create_sequences(train[features], train['target'], seq_length)
    X_test_seq, y_test_seq = create_sequences(valid[features], valid['target'], seq_length)

    # 데이터를 numpy 배열로 변환 후 텐서로 변환
    X_train = torch.tensor(X_train_seq, dtype=torch.float32)
    X_test = torch.tensor(X_test_seq, dtype=torch.float32)
    y_train = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1)  # 2D 텐서로 변환
    y_test = torch.tensor(y_test_seq, dtype=torch.float32).view(-1, 1)  # 2D 텐서로 변환

    # Bi-LSTM 모델 정의
    class BiLSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(BiLSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                batch_first=True, bidirectional=True)  # bidirectional=True로 설정
            self.fc = nn.Linear(hidden_size * 2, output_size)  # 양방향이므로 hidden_size * 2

        def forward(self, x):
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # (num_layers*2, batch_size, hidden_size)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # (num_layers*2, batch_size, hidden_size)
            out, _ = self.lstm(x, (h0, c0))  # LSTM에 입력 데이터 전달
            out = self.fc(out[:, -1, :])  # 마지막 시퀀스의 출력만 사용
            return out

    # 하이퍼파라미터 설정
    input_size = len(features)  # feature 개수
    hidden_size = 50  # hidden state 크기
    num_layers = 2  # LSTM 레이어 수
    output_size = 1  # 예측할 타겟 크기 (단일 값 예측)
    learning_rate = 0.001
    num_epochs = 100

    # 모델, 손실함수, 옵티마이저 정의
    model = BiLSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 모델 학습
    model.train()
    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 예측 수행
    model.eval()
    with torch.no_grad():
        pred = model(X_test).detach().numpy()
    
    # 성능 지표 계산 (RMSE, SMAPE)
    y_test_np = y_test.detach().numpy()  # y_test를 NumPy 배열로 변환
    rmse = np.sqrt(mean_squared_error(y_test_np, pred))
    smape_value = smape(y_test_np, pred)
    
    # 성능 오차 저장
    rmse_list.append(rmse)
    smape_list.append(smape_value)
    
    # 서브플롯 위치 지정
    ax = axs[idx // 6, idx % 6]
    
    # 예측 결과 서브플롯에 그리기
    ax.plot(train.index, train['target'], label='Train', color='blue')
    ax.plot(valid.index[seq_length:], valid['target'].iloc[seq_length:], label='Actual', color='green')
    ax.plot(valid.index[seq_length:], pred, label='Prediction', linestyle='--', color='orange')
    ax.set_title(f'Building {num} | RMSE: {rmse:.4f} | SMAPE: {smape_value:.4f}%', fontsize=10)
    ax.grid(True)

# 전체 서브플롯 레이아웃 조정 및 표시
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 상단에 제목 추가 공간 확보
plt.show()

# 성능 오차의 평균 계산
rmse_avg = np.mean(rmse_list)
smape_avg = np.mean(smape_list)

# 성능 오차의 평균 출력
print(f'Average RMSE for all buildings: {rmse_avg:.4f}')
print(f'Average SMAPE for all buildings: {smape_avg:.4f}%')
