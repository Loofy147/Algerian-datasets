from sqlalchemy.orm import sessionmaker
from .db.session import engine, Base
from .db.models import Company, Salary, Demographic, EconomicIndicator, SectoralData
from .data_loader import COMPANY_DATA_PATH, SALARY_DATA_PATH, load_and_clean_companies_from_csv, load_salaries_from_csv
from .services.ingestion import insert_demographic_data, insert_economic_indicators, insert_sectoral_data
import logging
import pandas as pd
import json
import os
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def seed_companies(db_session):
    """Seeds the companies table."""
    df = load_and_clean_companies_from_csv(COMPANY_DATA_PATH)
    if df.empty:
        logging.warning("Cleaned company data is empty, skipping seeding.")
        return

    companies_to_add = []
    processed_ids = set()
    for _, row in df.iterrows():
        company_id = str(row['company_id'])
        if company_id in processed_ids:
            continue

        if not db_session.query(Company).filter(Company.company_id == company_id).first():
            companies_to_add.append(Company(**row.to_dict()))
            processed_ids.add(company_id)

    if companies_to_add:
        db_session.add_all(companies_to_add)
        db_session.commit()
        logging.info(f"Successfully added {len(companies_to_add)} new companies.")
    else:
        logging.info("No new companies to add.")

def seed_salaries(db_session):
    """Seeds the salaries table."""
    # Delete existing salary data to ensure a clean slate
    db_session.query(Salary).delete()
    db_session.commit()

    df = load_salaries_from_csv(SALARY_DATA_PATH)
    if df.empty:
        logging.warning("Salary data is empty, skipping seeding.")
        return

    salaries_to_add = []
    for _, row in df.iterrows():
        salaries_to_add.append(Salary(**row.to_dict()))

    if salaries_to_add:
        db_session.add_all(salaries_to_add)
        db_session.commit()
        logging.info(f"Successfully added {len(salaries_to_add)} new salaries.")
    else:
        logging.info("No new salaries to add.")

def seed_database():
    """
    - Connects to the database and creates tables if they don't exist.
    - Populates the tables with the data, skipping duplicates.
    """
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db_session = Session()

    try:
        seed_companies(db_session)
        seed_salaries(db_session)
        
        # Seed new datasets
        base_path = Path(__file__).resolve().parent
        
        # Demographics
        demo_path = base_path / "algeria_data_platform" / "processed_demographics.csv"
        if demo_path.exists():
            df_demo = pd.read_csv(demo_path)
            insert_demographic_data(db_session, df_demo)
            
        # Economic Indicators
        econ_path = base_path / "algeria_data_platform" / "processed_economic_indicators.csv"
        if econ_path.exists():
            df_econ = pd.read_csv(econ_path)
            insert_economic_indicators(db_session, df_econ)
            
        # Sectoral Data
        sector_path = base_path / "algeria_data_platform" / "processed_sectoral_data.json"
        if sector_path.exists():
            with open(sector_path, "r") as f:
                sector_data = json.load(f)
                insert_sectoral_data(db_session, sector_data)
                
    except Exception as e:
        logging.error(f"An error occurred during database seeding: {e}")
        db_session.rollback()
    finally:
        db_session.close()

if __name__ == "__main__":
    logging.info("Starting database seeding process...")
    seed_database()
    logging.info("Database seeding process finished.")
