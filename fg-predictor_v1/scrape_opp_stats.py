import requests
from lxml import html
import polars as pl

def scrape_opponent_fg_stats(year):
    url = f"https://www.teamrankings.com/nfl/stat/opponent-field-goals-made-per-game?date={year}-02-12"
    response = requests.get(url)
    
    if response.status_code == 200:
        tree = html.fromstring(response.content)
        rows = tree.xpath('//table[@class="tr-table datatable scrollable"]/tbody/tr')
        
        print(f"Number of rows found for {year}: {len(rows)}")
        
        data = []
        for row in rows:
            team = row.xpath('./td[2]/a/text()')
            stats = row.xpath('./td/text()')
            
            if team and len(stats) >= 3:
                team = team[0].strip()
                fg_per_game = float(stats[2])
                data.append([team, fg_per_game, year])
                print(f"Scraped for {year}: {team} - {fg_per_game}")
            else:
                print(f"Failed to scrape row for {year}: {row.xpath('.//td/text()')}")
        
        return data
    else:
        print(f"Failed to retrieve the page for {year}. Status code: {response.status_code}")
        return None

def scrape_multiple_years(years):
    all_data = []
    
    for year in years:
        year_data = scrape_opponent_fg_stats(year)
        if year_data:
            all_data.extend(year_data)
    
    if all_data:
        df = pl.DataFrame({
            'Team': [row[0] for row in all_data],
            'Opp_FG_Per_Game': [row[1] for row in all_data],
            'Year': [row[2] for row in all_data]
        })
        return df
    else:
        return None

def main():
    years_to_scrape = ["2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", "2015"]
    df = scrape_multiple_years(years_to_scrape)
    
    if df is not None:
        csv_filename = "assets/opponent_fg_stats_multiple_years.csv"
        df.write_csv(csv_filename)
        print(f"Data saved to '{csv_filename}'")
        print(df.head())
    else:
        print("No data was scraped.")

if __name__ == "__main__":
    main()