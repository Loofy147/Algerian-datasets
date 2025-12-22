import pandas as pd
from sqlalchemy.orm import Session
from ..db.models import Company
import logging
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from pathlib import Path

# Build a dynamic path to the Great Expectations context
context_root_dir = Path(__file__).resolve().parent.parent.parent / "gx"
context = gx.get_context(context_root_dir=str(context_root_dir))

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    def __init__(self, message, validation_result):
        super().__init__(message)
        self.validation_result = validation_result

def validate_company_data(df: pd.DataFrame):
    """
    Validates company data from a pandas DataFrame using Great Expectations.
    Raises DataValidationError if validation fails.
    """
    batch_request = RuntimeBatchRequest(
        datasource_name="pandas_datasource",
        data_connector_name="runtime_data_connector",
        data_asset_name="company_ingestion",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"some_name_that_does_not_matter": "default_identifier"},
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="company_ingestion_suite",
    )
    validation_result = validator.validate()

    if not validation_result.success:
        raise DataValidationError("Data validation failed", validation_result)

def insert_company_data(db: Session, df: pd.DataFrame):
    """
    - Ingests company data into the database.
    - Skips duplicates based on the company ID.
    """
    incoming_ids = set(df['company_id'].astype(str).unique())

    existing_ids = {
        res[0] for res in db.query(Company.company_id).filter(Company.company_id.in_(incoming_ids))
    }

    new_ids = incoming_ids - existing_ids

    companies_to_add = []
    if new_ids:
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
        logging.info(f"Successfully added {len(companies_to_add)} new companies.")
    else:
        logging.info("No new companies to add.")
