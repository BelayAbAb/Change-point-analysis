import requests
import pandas as pd
from bs4 import BeautifulSoup
import os

def fetch_economic_data(start_year=1987, end_year=2022):
    """
    Fetches economic data (Country, GDP, GDP Growth, Interest Rate, Inflation Rate, 
    Jobless Rate, Gov. Budget, Debt/GDP, Current Account, Population) for each year 
    from Trading Economics and saves it to a CSV file.
    """
    url = 'https://tradingeconomics.com/'
    
    # Define headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    all_data = []

    # Loop through each year from start_year to end_year
    for year in range(start_year, end_year + 1):
        # Construct URL or modify the request to target the data for that year if necessary
        # Example URL: you might need to adjust this based on actual data structure
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch data for {year}: {response.status_code}")
            continue

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the relevant table
        table = soup.find('table', class_='table-hover')  
        if not table:
            print(f"No data table found for {year}.")
            continue

        # Extract data from the table
        rows = table.find_all('tr')[1:]  # Skip the header row
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 10:  
                country = cols[0].text.strip()
                gdp = cols[1].text.strip()
                gdp_growth = cols[2].text.strip()
                interest_rate = cols[3].text.strip()
                inflation_rate = cols[4].text.strip()
                jobless_rate = cols[5].text.strip()
                gov_budget = cols[6].text.strip()
                debt_gdp = cols[7].text.strip()
                current_account = cols[8].text.strip()
                population = cols[9].text.strip()
                all_data.append([country, gdp, gdp_growth, interest_rate, inflation_rate,
                                  jobless_rate, gov_budget, debt_gdp, current_account, population, year])

    # Create a DataFrame
    df = pd.DataFrame(all_data, columns=['Country', 'GDP', 'GDP Growth', 'Interest Rate', 
                                         'Inflation Rate', 'Jobless Rate', 'Gov. Budget', 
                                         'Debt/GDP', 'Current Account', 'Population', 'Year'])

    # Define the output directory
    output_dir = r"C:\Users\User\Desktop\10Acadamy\week 10\Change point analysis\New Data"
    os.makedirs(output_dir, exist_ok=True)

    # Save the data to a CSV file
    df.to_csv(os.path.join(output_dir, 'economic_data_full_years.csv'), index=False)

    print("Economic data fetched and saved successfully.")

if __name__ == "__main__":
    fetch_economic_data()
