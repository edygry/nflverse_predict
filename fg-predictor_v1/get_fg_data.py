import requests
from bs4 import BeautifulSoup
from lxml import html
import polars as pl
import re

def scrape_nfl_field_goal_stats(year):
    url = f"https://www.nfl.com/stats/team-stats/special-teams/field-goals/{year}/reg/all"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse with lxml for XPath
        tree = html.fromstring(response.content)
        
        # Parse with BeautifulSoup for other data
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table', class_='d3-o-table')
        headers = [th.text.strip() for th in table.find_all('th')]
        
        rows = []
        for i, tr in enumerate(table.find_all('tr')[1:], start=1):  # Skip header row
            # Try to extract team name using XPath, fall back to BeautifulSoup if it fails
            try:
                team_name = tree.xpath(f'//table[@class="d3-o-table"]/tbody/tr[{i}]/td[1]/div/div[3]/text()')[0].strip()
            except IndexError:
                # Fallback to BeautifulSoup
                team_name = tr.find('td').find('div', class_='d3-o-club-fullname').text.strip()
            
            # Extract other data using BeautifulSoup
            other_data = [td.text.strip() for td in tr.find_all('td')[1:]]
            
            full_row = [team_name] + other_data + [year]
            rows.append(full_row)
        
        # Add 'Year' to headers
        headers.append('Year')
        
        # Create a Polars DataFrame
        df = pl.DataFrame(rows, schema=headers)
        
        return df
    else:
        print(f"Failed to retrieve the page for year {year}. Status code: {response.status_code}")
        return None

def scrape_multiple_years(years):
    all_data = []
    
    for year in years:
        print(f"Scraping data for {year}...")
        df = scrape_nfl_field_goal_stats(year)
        if df is not None:
            all_data.append(df)
    
    if all_data:
        # Concatenate all DataFrames
        combined_df = pl.concat(all_data)
        
        # Save to CSV
        csv_filename = "assets/nfl_field_goal_stats_multiple_years.csv"
        combined_df.write_csv(csv_filename)
        print(f"Data saved to '{csv_filename}'")
        
        # Display the first few rows
        print(combined_df.head())
    else:
        print("No data was scraped.")

if __name__ == "__main__":
    # List of years to scrape
    years_to_scrape = ["2023", "2022", "2021","2020","2019","2018","2017","2016","2015"]  # Add or remove years as needed
    scrape_multiple_years(years_to_scrape)