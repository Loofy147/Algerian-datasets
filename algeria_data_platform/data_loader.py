import pandas as pd
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import Company
from pathlib import Path

# Define the path to the data directory, still needed for seeding script
DATA_DIR = Path(__file__).parent / "data"
COMPANY_DATA_PATH = DATA_DIR / "cnrc_sample_data.csv"

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
