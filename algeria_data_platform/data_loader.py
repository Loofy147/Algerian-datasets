import pandas as pd
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import Company
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the data directory
DATA_DIR = Path(__file__).parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
COMPANY_DATA_PATH = RAW_DATA_DIR / "cnrc_sample_data.csv"
SALARY_DATA_PATH = RAW_DATA_DIR / "salary_sample_data.csv"
DEMOGRAPHICS_DATA_PATH = RAW_DATA_DIR / "demographics.csv"
ECONOMIC_INDICATORS_DATA_PATH = RAW_DATA_DIR / "economic_indicators.csv"

def load_economic_indicators_from_csv(filepath: Path) -> pd.DataFrame:
    """
    Reads economic indicators data from a CSV file into a pandas DataFrame.
    """
    if not filepath.exists():
        logging.error(f"Data file not found at: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    return df

def load_demographics_from_csv(filepath: Path) -> pd.DataFrame:
    """
    Reads demographics data from a CSV file into a pandas DataFrame.
    """
    if not filepath.exists():
        logging.error(f"Data file not found at: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    return df

def load_salaries_from_csv(filepath: Path) -> pd.DataFrame:
    """
    Reads salary data from a CSV file into a pandas DataFrame.
    """
    if not filepath.exists():
        logging.error(f"Data file not found at: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    return df

def clean_company_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and prepares the company DataFrame by renaming columns,
    handling missing values, and adding required fields.
    """
    # Rename columns to match the database model
    df.rename(columns={'company_name': 'legal_name'}, inplace=True)

    # Add 'trade_name' and set a default 'status'
    if 'trade_name' not in df.columns:
        df['trade_name'] = ""
    df['status'] = 'Active'

    # Data cleaning and validation
    df.dropna(subset=['company_id', 'legal_name'], inplace=True)
    df['company_id'] = df['company_id'].astype(str)
    df = df[df['legal_name'] != 'Unknown Company']

    # Drop columns that are not in the Company model
    df = df[['company_id', 'legal_name', 'trade_name', 'status']]

    return df

def load_and_clean_companies_from_csv(filepath: Path) -> pd.DataFrame:
    """
    Reads company data from a CSV, then cleans and prepares it for the database.
    """
    if not filepath.exists():
        logging.error(f"Data file not found at: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df = clean_company_data(df)
    return df

def get_all_companies_as_df(db: Session) -> pd.DataFrame:
    """
    Retrieves all company records from the database and returns them
    as a pandas DataFrame.
    """
    query = db.query(Company)
    df = pd.read_sql(query.statement, db.bind)
    return df

if __name__ == '__main__':
    # A simple script to test the new database-backed function
    db_session = SessionLocal()
    try:
        company_data = get_all_companies_as_df(db_session)
        if not company_data.empty:
            print("Successfully loaded data from the database:")
            print(company_data.to_string())
            print(f"\nTotal rows loaded: {len(company_data)}")
        else:
            print("No data in the database.")
    finally:
        db_session.close()
