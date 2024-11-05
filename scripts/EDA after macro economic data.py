import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
oil_data_path = r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\New Data\BrentOilPrices.csv"
economic_data_path = r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\New Data\economic_data_full_years.csv"

# Reading the datasets
oil_df = pd.read_csv(oil_data_path, parse_dates=['Date'], dayfirst=True)
economic_df = pd.read_csv(economic_data_path)

# Displaying the first few rows to understand the structure of the data
print("Oil Data Head:\n", oil_df.head())
print("\nEconomic Data Head:\n", economic_df.head())

# Data Cleaning - Checking for missing values
print("\nOil Data Missing Values:\n", oil_df.isnull().sum())
print("\nEconomic Data Missing Values:\n", economic_df.isnull().sum())

# Handle missing values if necessary (you can either drop or fill them, depending on the requirement)
# Example: Drop rows with missing values
oil_df = oil_df.dropna()
economic_df = economic_df.dropna()

# Basic Summary Statistics for Oil Data and Economic Data
print("\nOil Data Summary Statistics:\n", oil_df.describe())
print("\nEconomic Data Summary Statistics:\n", economic_df.describe())

# EDA: Plotting the Time Series of Oil Prices
plt.figure(figsize=(10, 6))
plt.plot(oil_df['Date'], oil_df['Price'], label='Brent Oil Price', color='blue')
plt.title('Brent Oil Price Time Series')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report\oil_price_time_series.jpg")
plt.show()

# EDA: Correlation Heatmap for Economic Data
# First, select only numeric columns to compute correlation
economic_numeric_df = economic_df.select_dtypes(include=['float64', 'int64'])

# Calculate correlations only for numeric columns
plt.figure(figsize=(10, 8))
economic_corr = economic_numeric_df.corr()  # Calculate correlations
sns.heatmap(economic_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Economic Data')
plt.tight_layout()
plt.savefig(r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report\economic_correlation_heatmap.jpg")
plt.show()

# EDA: Distribution of Key Economic Indicators
# Plotting distribution for GDP, Inflation Rate, Jobless Rate
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.histplot(economic_df['GDP'], kde=True, color='purple', bins=20)
plt.title('GDP Distribution')

plt.subplot(2, 2, 2)
sns.histplot(economic_df['Inflation Rate'], kde=True, color='green', bins=20)
plt.title('Inflation Rate Distribution')

plt.subplot(2, 2, 3)
sns.histplot(economic_df['Jobless Rate'], kde=True, color='red', bins=20)
plt.title('Jobless Rate Distribution')

plt.tight_layout()
plt.savefig(r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report\economic_distribution_plots.jpg")
plt.show()

# EDA: Boxplot for Oil Prices (Checking for outliers)
plt.figure(figsize=(8, 6))
sns.boxplot(x=oil_df['Price'], color='orange')
plt.title('Boxplot of Oil Prices')
plt.xlabel('Price (USD)')
plt.tight_layout()
plt.savefig(r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report\oil_price_boxplot.jpg")
plt.show()

# EDA: Correlation between Oil Prices and Economic Indicators (Using Yearly Data)
# Merge the oil data with economic data on Year
oil_df['Year'] = oil_df['Date'].dt.year
merged_df = pd.merge(oil_df, economic_df, on='Year', how='left')

# Select only numeric columns for the merged data
merged_numeric_df = merged_df.select_dtypes(include=['float64', 'int64'])

# Correlation plot between oil prices and economic indicators
plt.figure(figsize=(10, 6))
correlation = merged_numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation between Oil Prices and Economic Indicators')
plt.tight_layout()
plt.savefig(r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\Report\oil_economic_correlation.jpg")
plt.show()
