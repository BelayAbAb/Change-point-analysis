import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the output folder path
output_folder = r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report"

# Step 1: Load the Brent Oil Price data (Daily)
oil_data_path = r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\New Data\BrentOilPrices.csv"
economic_data_path = r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\New Data\economic_data_full_years.csv"

# Load oil price data from CSV
oil_df = pd.read_csv(oil_data_path, parse_dates=['Date'], dayfirst=True)

# Extract year from the 'Date' column (ensure it is an integer)
oil_df['Year'] = oil_df['Date'].dt.year

# Load economic data from CSV
economic_df = pd.read_csv(economic_data_path)

# Ensure the 'Year' column in economic_df is of integer type
economic_df['Year'] = economic_df['Year'].astype(int)

# Step 2: Data Merging (Aligning the economic data with oil price data)
# Merge both DataFrames on the 'Year' column
merged_df = pd.merge(oil_df, economic_df, on='Year', how='left')

# Step 3: Check for Stationarity of the Oil Price Data (ADF Test)
def adf_test(series, output_path):
    result = adfuller(series)
    with open(output_path, 'w') as f:
        f.write(f'ADF Statistic: {result[0]}\n')
        f.write(f'p-value: {result[1]}\n')
        if result[1] <= 0.05:
            f.write("Series is stationary\n")
        else:
            f.write("Series is not stationary\n")

adf_test(merged_df['Price'], f'{output_folder}\\ADF_test_result.txt')

# Step 4: Apply ARIMA Model for Oil Price Forecasting
# ARIMA Model (Adjust the order as needed based on ACF/PACF analysis)
model_arima = ARIMA(merged_df['Price'], order=(5, 1, 0))  # Adjust order
model_arima_fit = model_arima.fit()

# Save ARIMA model summary to a file
with open(f'{output_folder}\\ARIMA_model_summary.txt', 'w') as f:
    f.write(str(model_arima_fit.summary()))

# ARIMA Forecasting (Next 12 days)
forecast_arima = model_arima_fit.forecast(steps=12)

# ARIMA Forecast Plot
plt.figure(figsize=(10, 6))
plt.plot(merged_df.index, merged_df['Price'], label='Observed')
plt.plot(pd.date_range(merged_df.index[-1], periods=13, freq='D')[1:], forecast_arima, label='Forecast', color='red')
plt.title('ARIMA Model Forecast for Oil Prices')
plt.xlabel('Date')
plt.ylabel('Brent Oil Price (USD)')
plt.legend()
plt.savefig(f'{output_folder}\\ARIMA_forecast_plot.jpg', format='jpg')  # Save as .jpg
plt.close()

# Step 5: Apply VAR Model for Multivariate Time Series Analysis
var_data = merged_df[['Price', 'GDP', 'Inflation Rate', 'Jobless Rate']].dropna()
train_size = int(len(var_data) * 0.8)
train, test = var_data[:train_size], var_data[train_size:]

# Fit VAR model
model_var = VAR(train)
results_var = model_var.fit(5)

# Save VAR model summary to a file
with open(f'{output_folder}\\VAR_model_summary.txt', 'w') as f:
    f.write(str(results_var.summary()))

# VAR Forecasting
forecast_var = results_var.forecast(train.values[-5:], steps=len(test))
forecast_var_df = pd.DataFrame(forecast_var, index=test.index, columns=test.columns)

# VAR Forecast Plot
plt.figure(figsize=(10, 6))
plt.plot(test['Price'], label='Observed')
plt.plot(forecast_var_df['Price'], label='Forecast', color='red')
plt.title('VAR Model Forecast for Oil Prices')
plt.xlabel('Date')
plt.ylabel('Brent Oil Price (USD)')
plt.legend()
plt.savefig(f'{output_folder}\\VAR_forecast_plot.jpg', format='jpg')  # Save as .jpg
plt.close()

# Step 6: Model Evaluation (ARIMA and VAR)
# ARIMA Model Evaluation
arima_rmse = np.sqrt(mean_squared_error(merged_df['Price'][-12:], forecast_arima))
arima_mae = mean_absolute_error(merged_df['Price'][-12:], forecast_arima)
arima_r2 = r2_score(merged_df['Price'][-12:], forecast_arima)

# Save ARIMA evaluation metrics to file
with open(f'{output_folder}\\ARIMA_model_evaluation.txt', 'w') as f:
    f.write(f"ARIMA RMSE: {arima_rmse}\n")
    f.write(f"ARIMA MAE: {arima_mae}\n")
    f.write(f"ARIMA R2: {arima_r2}\n")

# VAR Model Evaluation
var_rmse = np.sqrt(mean_squared_error(test['Price'], forecast_var_df['Price']))
var_mae = mean_absolute_error(test['Price'], forecast_var_df['Price'])
var_r2 = r2_score(test['Price'], forecast_var_df['Price'])

# Save VAR evaluation metrics to file
with open(f'{output_folder}\\VAR_model_evaluation.txt', 'w') as f:
    f.write(f"VAR RMSE: {var_rmse}\n")
    f.write(f"VAR MAE: {var_mae}\n")
    f.write(f"VAR R2: {var_r2}\n")
