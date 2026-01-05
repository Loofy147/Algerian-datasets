import pandas as pd
import json
import os

def process_data():
    # 1. Process Economic Indicators
    econ_df = pd.read_csv("comprehensive_economic_indicators.csv")
    # Add some derived metrics
    econ_df['gdp_per_capita_growth'] = econ_df['gdp_growth'] - 1.5 # Simplified
    econ_df.to_csv("../processed_economic_indicators.csv", index=False)
    print("Processed economic indicators.")

    # 2. Process Demographic Data (Sample from research)
    demographic_data = [
        {"wilaya_code": "16", "wilaya_name": "Alger", "population_2024": 3500000, "area_km2": 1190, "urbanization_rate": 0.95},
        {"wilaya_code": "31", "wilaya_name": "Oran", "population_2024": 1500000, "area_km2": 2114, "urbanization_rate": 0.85},
        {"wilaya_code": "25", "wilaya_name": "Constantine", "population_2024": 1000000, "area_km2": 2187, "urbanization_rate": 0.80},
        {"wilaya_code": "06", "wilaya_name": "Béjaïa", "population_2024": 950000, "area_km2": 3268, "urbanization_rate": 0.65},
        {"wilaya_code": "19", "wilaya_name": "Sétif", "population_2024": 1600000, "area_km2": 6504, "urbanization_rate": 0.70}
    ]
    demo_df = pd.DataFrame(demographic_data)
    demo_df['density_per_km2'] = demo_df['population_2024'] / demo_df['area_km2']
    demo_df.to_csv("../processed_demographics.csv", index=False)
    print("Processed demographics.")

    # 3. Process Sectoral Data
    sectoral_data = {
        "sectors": [
            {"name": "Agriculture", "value_usd_billions": 24, "growth_rate": 4.5},
            {"name": "Energy", "value_usd_billions": 45, "growth_rate": 2.1},
            {"name": "Real Estate", "value_usd_billions": 960, "growth_rate": 9.1},
            {"name": "Tourism", "value_usd_billions": 3.5, "growth_rate": 6.8}
        ]
    }
    with open("../processed_sectoral_data.json", "w") as f:
        json.dump(sectoral_data, f, indent=2)
    print("Processed sectoral data.")

if __name__ == "__main__":
    process_data()
