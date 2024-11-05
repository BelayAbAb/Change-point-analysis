from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the data (assuming CSV files)
oil_data_path = r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\New Data\BrentOilPrices.csv"
economic_data_path = r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\New Data\economic_data_full_years.csv"

# Load the oil price and economic data
oil_df = pd.read_csv(oil_data_path, parse_dates=['Date'], dayfirst=True)
economic_df = pd.read_csv(economic_data_path)

# Convert economic data to numeric for correlation
economic_numeric_df = economic_df.select_dtypes(include=['float64', 'int64'])

# Home page route to load the initial view
@app.route('/')
def home():
    return render_template('index.html')

# API routes
@app.route('/api/oil-prices', methods=['GET'])
def get_oil_prices():
    oil_prices = oil_df[['Date', 'Price']].to_dict(orient='records')
    return jsonify(oil_prices)

@app.route('/api/economic-indicators', methods=['GET'])
def get_economic_indicators():
    economic_indicators = economic_df.to_dict(orient='records')
    return jsonify(economic_indicators)

@app.route('/api/correlation', methods=['GET'])
def get_correlation():
    correlation_matrix = economic_numeric_df.corr()
    correlation_dict = correlation_matrix.to_dict()
    return jsonify(correlation_dict)

@app.route('/api/events', methods=['GET'])
def get_events():
    # Assuming events data is available, if not, use a placeholder
    events = [
        {"date": "1987-05-01", "event": "US Sanctions on Iran", "price_impact": "Increase"},
        {"date": "1987-06-01", "event": "Oil Crisis", "price_impact": "Decrease"}
    ]
    return jsonify(events)

if __name__ == '__main__':
    app.run(debug=True)
