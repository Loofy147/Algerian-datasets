import pandas as pd
import numpy as np

def generate_economic_data():
    years = list(range(2015, 2027))
    data = {
        "year": years,
        "gdp_growth": [3.7, 3.2, 1.4, 1.2, 0.8, -4.9, 3.4, 3.2, 4.1, 3.8, 3.0, 2.9],
        "inflation": [4.8, 6.4, 5.6, 4.3, 2.0, 2.4, 7.2, 9.3, 9.3, 7.5, 5.0, 3.9],
        "oil_price_avg": [52, 43, 54, 71, 64, 41, 70, 99, 82, 80, 75, 72],
        "population_millions": [39.7, 40.6, 41.4, 42.2, 43.0, 43.8, 44.6, 45.4, 46.2, 47.0, 47.8, 48.6],
        "unemployment_rate": [11.2, 10.5, 11.7, 11.7, 11.4, 12.5, 12.0, 11.5, 11.0, 10.8, 10.5, 10.2]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("comprehensive_economic_indicators.csv", index=False)
    print("Generated comprehensive_economic_indicators.csv")

if __name__ == "__main__":
    generate_economic_data()
