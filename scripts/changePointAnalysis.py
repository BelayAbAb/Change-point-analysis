import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# File path for the data
file_path = r'C:\Users\User\Desktop\10Acadamy\week 10\data\BrentOilPrices.csv'

# Load data from CSV
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])  # Automatically infer the date format

# A. Line plot of Brent Oil Prices
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], marker='o')
plt.title('Brent Oil Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report\brent_oil_prices.jpg')
plt.close()

# B. CUSUM Analysis
mean_price = df['Price'].mean()
cusum = np.cumsum(df['Price'] - mean_price)

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], cusum, marker='o')
plt.axhline(0, color='red', linestyle='--')
plt.title('CUSUM Analysis of Brent Oil Prices')
plt.xlabel('Date')
plt.ylabel('CUSUM')
plt.grid()
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report\cusum_analysis.jpg')
plt.close()

# C. Posterior Distribution of Mean Price
posterior_samples = np.random.normal(loc=mean_price, scale=1, size=1000)

plt.figure(figsize=(10, 6))
plt.hist(posterior_samples, bins=30, density=True, alpha=0.5, color='blue')
plt.title('Posterior Distribution of Mean Price')
plt.xlabel('Price')
plt.ylabel('Density')
plt.grid()
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report\posterior_distribution.jpg')
plt.close()

# D. Brent Oil Prices with Change Point (example change point at index 30)
change_point_index = 30
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], marker='o')
plt.axvline(df['Date'][change_point_index], color='red', linestyle='--', label='Change Point')
plt.title('Brent Oil Prices with Change Point')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report\change_point.jpg')
plt.close()

# E. Brent Oil Prices with Detected Change Points
detected_change_points = [30, 50]  # example indices of detected change points
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], marker='o')
for point in detected_change_points:
    plt.axvline(df['Date'][point], color='red', linestyle='--')
plt.title('Brent Oil Prices with Detected Change Points')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report\detected_change_points.jpg')
plt.close()
