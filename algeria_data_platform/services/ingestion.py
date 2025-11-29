import pandas as pd
from sqlalchemy.orm import Session
from ..models import Company
import logging

def ingest_company_data(db: Session, df: pd.DataFrame):
    """
    - Ingests company data from a pandas DataFrame into the database.
    - Skips duplicates based on the company ID.
    - Logs the number of new companies added.
    """
    # Get a set of all company IDs from the DataFrame
    incoming_ids = set(df['company_id'].astype(str).unique())

    # Get a set of all company IDs that already exist in the database
    existing_ids = {
        res[0] for res in db.query(Company.company_id).filter(Company.company_id.in_(incoming_ids))
    }

    # Determine which companies are new
    new_ids = incoming_ids - existing_ids

    companies_to_add = []
    if new_ids:
        # Filter the DataFrame to only include new companies
        new_companies_df = df[df['company_id'].astype(str).isin(new_ids)].drop_duplicates(subset=['company_id'])

        for _, row in new_companies_df.iterrows():
            companies_to_add.append(Company(
                company_id=str(row['company_id']),
                legal_name=row.get('legal_name'),
                trade_name=row.get('trade_name'),
                status=row.get('status')
            ))

    if companies_to_add:
        db.add_all(companies_to_add)
        db.commit()
        logging.info(f"Successfully added {len(companies_to_add)} new companies to the database.")
    else:
        logging.info("No new companies to add.")
