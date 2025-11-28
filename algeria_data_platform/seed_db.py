import pandas as pd
from sqlalchemy.orm import sessionmaker
from .database import engine, Base
from .models import Company
from .data_loader import COMPANY_DATA_PATH
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def seed_database():
    """
    - Reads company data from the sample CSV file.
    - Connects to the database and creates the 'companies' table if it doesn't exist.
    - Populates the table with the data, skipping duplicates.
    """
    # Create the table in the database
    Base.metadata.create_all(bind=engine)

    # Create a new session
    Session = sessionmaker(bind=engine)
    db_session = Session()

    try:
        # Load data from CSV
        if not COMPANY_DATA_PATH.exists():
            logging.error(f"Data file not found at: {COMPANY_DATA_PATH}")
            return

        df = pd.read_csv(COMPANY_DATA_PATH)

        # Keep track of companies to add and IDs processed from the CSV
        companies_to_add = []
        processed_ids = set()

        for _, row in df.iterrows():
            company_id = str(row['company_id'])

            # Skip if the company ID has already been processed from the CSV
            if company_id in processed_ids:
                continue

            # Check if company already exists in the database to prevent duplicates
            exists = db_session.query(Company).filter(Company.company_id == company_id).first()
            if not exists:
                company = Company(
                    company_id=company_id,
                    legal_name=row.get('legal_name'),
                    trade_name=row.get('trade_name'),
                    status=row.get('status')
                )
                companies_to_add.append(company)
                processed_ids.add(company_id)

        if companies_to_add:
            db_session.add_all(companies_to_add)
            db_session.commit()
            logging.info(f"Successfully added {len(companies_to_add)} new companies to the database.")
        else:
            logging.info("No new companies to add.")

    except Exception as e:
        logging.error(f"An error occurred during database seeding: {e}")
        db_session.rollback()
    finally:
        db_session.close()

if __name__ == "__main__":
    logging.info("Starting database seeding process...")
    seed_database()
    logging.info("Database seeding process finished.")
