import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the data
file_path = r'C:\Users\User\Desktop\10Acadamy\week 10\data\BrentOilPrices.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 1. Model Building
# a. ARIMA Model
arima_model = ARIMA(df['Price'], order=(5, 1, 0))  
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
history = model.fit(X_train, y_train, epochs=100, batch_size=32)

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
lstm_predictions = scaler.inverse_transform(lstm_predictions).flatten()  # Rescale and flatten

# Ensure the lengths match for plotting
if len(lstm_predictions) != len(y_test):
    print(f"Length mismatch: lstm_predictions={len(lstm_predictions)}, y_test={len(y_test)}")
else:
    # 3. Generate Reports as JPG
    output_path = r'C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report'

    # Model Building Outputs
    # a. ARIMA Model Outputs
    plt.figure(figsize=(10, 6))
    plt.title('ARIMA Model Summary')
    plt.text(0.1, 0.9, f"Coefficients:\n{arima_results.params}", fontsize=10)
    plt.text(0.1, 0.6, f"AIC: {arima_results.aic}", fontsize=10)
    plt.text(0.1, 0.5, f"BIC: {arima_results.bic}", fontsize=10)
    plt.axis('off')
    plt.savefig(f'{output_path}\\arima_model_summary.jpg')
    plt.close()

    # Forecast Values for ARIMA
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(test):], test, label='Actual Prices')
    plt.plot(df.index[-len(test):], arima_forecast, label='ARIMA Forecast', color='orange')
    plt.title('ARIMA Forecast vs Actual Prices')
    plt.legend()
    plt.savefig(f'{output_path}\\arima_forecast.jpg')
    plt.close()

    # b. GARCH Model Outputs
    plt.figure(figsize=(10, 6))
    plt.title('GARCH Model Summary')
    plt.text(0.1, 0.9, f"Parameters:\n{garch_results.params}", fontsize=10)
    plt.axis('off')
    plt.savefig(f'{output_path}\\garch_model_summary.jpg')
    plt.close()

    # Forecasted Volatility for GARCH
    garch_volatility = np.sqrt(garch_results.conditional_volatility)
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(garch_volatility):], garch_volatility, label='Forecasted Volatility', color='green')
    plt.title('GARCH Forecasted Volatility')
    plt.legend()
    plt.savefig(f'{output_path}\\garch_forecasted_volatility.jpg')
    plt.close()

    # c. LSTM Model Outputs
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('LSTM Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_path}\\lstm_training_loss.jpg')
    plt.close()

    # Predicted Values for LSTM
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(test):], test, label='Actual Prices')
    plt.plot(df.index[-len(test):], lstm_predictions, label='LSTM Predictions', color='orange')
    plt.title('LSTM Predictions vs Actual Prices')
    plt.legend()
    plt.savefig(f'{output_path}\\lstm_predictions.jpg')
    plt.close()
