import pandas as pd
from datetime import datetime

# Define the indicators you want to analyze
indicators = {
    "Economic Indicators": {
        "GDP": "Analyze the correlation between GDP growth rates of major economies and oil prices.",
        "Inflation Rates": "Examine how inflation in key economies impacts oil demand and prices.",
        "Unemployment Rates": "Investigate the relationship between unemployment rates and oil consumption patterns.",
        "Exchange Rates": "Assess the effect of currency fluctuations, especially the USD, on oil prices."
    },
    "Technological Changes": {
        "Advancements in Extraction Technologies": "Study the impact of technologies like hydraulic fracturing (fracking) and deep-sea drilling on oil supply.",
        "Renewable Energy Developments": "Analyze how growth in renewable energy sources affects oil demand and prices.",
        "Efficiency Improvements": "Evaluate how improvements in fuel efficiency and alternative energy usage influence oil consumption."
    },
    "Political and Regulatory Factors": {
        "Environmental Regulations": "Investigate how stricter environmental regulations and carbon pricing affect oil production and prices.",
        "Trade Policies": "Study the impact of trade agreements, tariffs, and embargoes on oil markets."
    }
}

# Define the years for analysis
years = list(range(1985, 2024))  # From 1985 to 2023 (39 years)

# Function to simulate data extraction and analysis
def extract_and_analyze_data(indicators, years):
    data = []
    
    for year in years:
        # Format the date as "20-May-87"
        formatted_date = datetime(year, 5, 20).strftime("%d-%b-%y")
        
        for category, desc in indicators.items():
            for indicator, analysis in desc.items():
                # Simulate data extraction (replace with actual data retrieval logic)
                simulated_value = 0  # Replace with actual data retrieval logic
                
                # Append extracted data to the list
                data.append({
                    "Date": formatted_date,
                    "Category": category,
                    "Indicator": indicator,
                    "Analysis": analysis,
                    "Value": simulated_value  # Replace with actual values
                })

    # Convert the data into a DataFrame
    df = pd.DataFrame(data)
    
    return df

# Extract and analyze the data
df_results = extract_and_analyze_data(indicators, years)

# Save results to CSV
output_file = r'C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\New Data\economic_indicators_analysis.csv'
df_results.to_csv(output_file, index=False)

print(f"Data successfully saved to {output_file}")
