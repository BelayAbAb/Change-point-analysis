import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load the data
file_path = r'C:\Users\User\Desktop\10Acadamy\week 10\data\BrentOilPrices.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 1. Model Building
# a. ARIMA Model
arima_model = ARIMA(df['Price'], order=(5, 1, 0))  # Example order
arima_results = arima_model.fit()

# b. GARCH Model
garch_model = arch_model(df['Price'], vol='Garch', p=1, q=1)
garch_results = garch_model.fit()

# c. LSTM Model
# Preprocessing for LSTM
data = df['Price'].values
data = data.reshape((len(data), 1))  # Reshape for LSTM
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:]

# Scaling the data (important for LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Create dataset for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X_train, y_train = create_dataset(train_scaled, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# 2. Model Evaluation
# a. ARIMA Evaluation
arima_forecast = arima_results.forecast(steps=len(test))
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
arima_mae = mean_absolute_error(test, arima_forecast)

# b. GARCH Evaluation
garch_forecast = garch_results.forecast(horizon=len(test))

# c. LSTM Evaluation
X_test, y_test = create_dataset(test_scaled, time_step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
lstm_predictions = model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)  # Rescale back

lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
lstm_mae = mean_absolute_error(y_test, lstm_predictions)

# 3. Generate Reports as JPG
output_path = r'C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report'

# Model Evaluation Metrics
plt.figure(figsize=(8, 6))
plt.bar(['ARIMA RMSE', 'ARIMA MAE', 'LSTM RMSE', 'LSTM MAE'],
        [arima_rmse, arima_mae, lstm_rmse, lstm_mae], color=['blue', 'blue', 'orange', 'orange'])
plt.title('Model Evaluation Metrics')
plt.ylabel('Error')
plt.grid()
plt.savefig(f'{output_path}\\model_evaluation_metrics.jpg')
plt.close()

# Model Comparison
better_model = 'ARIMA' if arima_rmse < lstm_rmse else 'LSTM'
plt.figure(figsize=(8, 6))
plt.text(0.5, 0.5, f'{better_model} model performs better based on RMSE.', fontsize=12, ha='center')
plt.axis('off')
plt.title('Model Comparison Result')
plt.savefig(f'{output_path}\\model_comparison_result.jpg')
plt.close()

# Insights
plt.figure(figsize=(8, 6))
insight_text = ("Actionable Insights:\n"
                f"- Based on the forecasts, it is recommended to monitor Brent oil prices closely.\n"
                f"- Key influencing factors include global economic indicators and geopolitical events.")
plt.text(0.5, 0.5, insight_text, fontsize=12, ha='center')
plt.axis('off')
plt.title('Insights Generated')
plt.savefig(f'{output_path}\\insights_generated.jpg')
plt.close()
