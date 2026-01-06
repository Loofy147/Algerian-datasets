import pandas as pd
from sqlalchemy.orm import Session
from ..db.models import Company, Salary
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

INGESTION_REGISTRY = {
    "companies": {
        "model": Company,
        "expectation_suite_name": "company_ingestion_suite",
        "pk": "company_id",
        "column_mapping": {
            "company_id": "company_id",
            "legal_name": "legal_name",
            "trade_name": "trade_name",
            "status": "status",
        },
    },
    "salaries": {
        "model": Salary,
        "expectation_suite_name": "salary_ingestion_suite",
        "pk": None, # No business key to check for duplicates
        "datetime_columns": ["scraped_at"],
        "column_mapping": {
            "job_title": "job_title",
            "min_salary_dzd": "min_salary_dzd",
            "max_salary_dzd": "max_salary_dzd",
            "currency": "currency",
            "period": "period",
            "source": "source",
            "scraped_at": "scraped_at",
        },
    },
}

def validate_data(data_type: str, df: pd.DataFrame):
    """
    Validates data from a pandas DataFrame using Great Expectations based on the data type.
    Raises DataValidationError if validation fails.
    """
    if data_type not in INGESTION_REGISTRY:
        raise ValueError(f"Unknown data type: {data_type}")

    config = INGESTION_REGISTRY[data_type]
    expectation_suite_name = config["expectation_suite_name"]

    batch_request = RuntimeBatchRequest(
        datasource_name="pandas_datasource",
        data_connector_name="runtime_data_connector",
        data_asset_name=f"{data_type}_ingestion",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"some_name_that_does_not_matter": "default_identifier"},
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=expectation_suite_name,
    )
    validation_result = validator.validate()

    if not validation_result.success:
        raise DataValidationError("Data validation failed", validation_result)


def insert_data(db: Session, data_type: str, df: pd.DataFrame):
    """
    - Ingests data into the database based on the data type.
    - Skips duplicates based on the primary key defined in the registry.
    """
    if data_type not in INGESTION_REGISTRY:
        raise ValueError(f"Unknown data type: {data_type}")

    config = INGESTION_REGISTRY[data_type]
    model = config["model"]
    pk_column = config.get("pk")
    column_mapping = config["column_mapping"]
    datetime_columns = config.get("datetime_columns", [])

    # Rename columns to match model attributes
    df = df.rename(columns=column_mapping)

    # Convert datetime columns
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    records_to_add = []

    if pk_column:
        # Logic to skip duplicates
        incoming_ids = set(df[pk_column].astype(str).unique())

        existing_ids = {
            str(res[0]) for res in db.query(getattr(model, pk_column)).filter(getattr(model, pk_column).in_(incoming_ids))
        }

        new_ids = incoming_ids - existing_ids

        if new_ids:
            new_records_df = df[df[pk_column].astype(str).isin(new_ids)].drop_duplicates(subset=[pk_column])

            for _, row in new_records_df.iterrows():
                # Filter for columns that exist in the model
                model_columns = model.__table__.columns.keys()
                record_data = {col: row.get(col) for col in column_mapping.values() if col in model_columns}
                records_to_add.append(model(**record_data))
    else:
        # No primary key check, insert all records
        for _, row in df.iterrows():
            model_columns = model.__table__.columns.keys()
            record_data = {col: row.get(col) for col in column_mapping.values() if col in model_columns}
            records_to_add.append(model(**record_data))


    if records_to_add:
        db.add_all(records_to_add)
        db.commit()
        logging.info(f"Successfully added {len(records_to_add)} new {data_type} records.")
    else:
        logging.info(f"No new {data_type} records to add.")
