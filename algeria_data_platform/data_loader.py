import pandas as pd
from pathlib import Path
import numpy as np

# Define the path to the data directory
DATA_DIR = Path(__file__).parent / "data"
COMPANY_DATA_PATH = DATA_DIR / "cnrc_sample_data.csv"

def load_and_clean_company_data() -> pd.DataFrame:
    """
    Loads the company registry data from the seed CSV file and performs
    a basic quality check.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned company data.
    """
    if not COMPANY_DATA_PATH.exists():
        return pd.DataFrame()

    # Load the raw "Bronze" data
    df = pd.read_csv(COMPANY_DATA_PATH)

    # --- Basic Data Quality Check (Transition to "Silver") ---
    # 1. Drop rows where the primary identifier is missing.
    cleaned_df = df.dropna(subset=['company_id'])

    # 2. Ensure company_id is a clean string (int -> str)
    cleaned_df = cleaned_df.astype({'company_id': 'int64'}).astype({'company_id': 'str'})

    # 3. Replace NaN values with None for JSON compatibility
    final_df = cleaned_df.replace({np.nan: None})

    return final_df

if __name__ == '__main__':
    # A simple script to test the loader function
    company_data = load_and_clean_company_data()
    print("Successfully loaded and cleaned data:")
    print(company_data.to_string())
    print(f"\nOriginal row count: {len(pd.read_csv(COMPANY_DATA_PATH))}")
    print(f"Cleaned row count: {len(company_data)}")
