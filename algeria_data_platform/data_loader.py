import pandas as pd
from sqlalchemy.orm import Session
from .db.session import SessionLocal
from .db.models import Company
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the data directory, still needed for seeding script
DATA_DIR = Path(__file__).parent / "data"
COMPANY_DATA_PATH = DATA_DIR / "cnrc_sample_data.csv"

def load_companies_from_csv(filepath: Path) -> pd.DataFrame:
    """
    Reads company data from a CSV file into a pandas DataFrame.

    Args:
        filepath (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the company data, or an empty
                      DataFrame if the file is not found.
    """
    if not filepath.exists():
        logging.error(f"Data file not found at: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    return df

def get_all_companies_as_df(db: Session) -> pd.DataFrame:
    """
    Retrieves all company records from the database and returns them
    as a pandas DataFrame.

    Args:
        db (Session): The SQLAlchemy database session.

    Returns:
        pd.DataFrame: A DataFrame containing all company data.
    """
    query = db.query(Company)
    df = pd.read_sql(query.statement, db.bind)
    return df

if __name__ == '__main__':
    # A simple script to test the new database-backed function
    db_session = SessionLocal()
    try:
        company_data = get_all_companies_as_df(db_session)
        print("Successfully loaded data from the database:")
        print(company_data.to_string())
        print(f"\nTotal rows loaded: {len(company_data)}")
    finally:
        db_session.close()
