import pandas as pd
from sqlalchemy.orm import Session
from ..db.models import Company, Demographic, EconomicIndicator, SectoralData
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
    - Handles pandas NaN/NaT values for database compatibility.
    """
    incoming_ids = set(df['company_id'].astype(str).unique())

    existing_ids = {
        res[0] for res in db.query(Company.company_id).filter(Company.company_id.in_(incoming_ids))
    }

    new_ids = incoming_ids - existing_ids

    companies_to_add = []
    if new_ids:
        # Filter for new companies and drop duplicates
        new_companies_df = df[df['company_id'].astype(str).isin(new_ids)].drop_duplicates(subset=['company_id'])

        # Replace pandas' null types with None for SQLAlchemy
        new_companies_df = new_companies_df.astype(object).where(pd.notnull(new_companies_df), None)

        for _, row in new_companies_df.iterrows():
            companies_to_add.append(Company(
                company_id=str(row['company_id']),
                legal_name=row.get('legal_name'),
                trade_name=row.get('trade_name'),
                status=row.get('status'),
                capital_amount_dzd=row.get('capital_amount_dzd'),
                registration_date=row.get('registration_date'),
                wilaya=row.get('wilaya'),
                legal_form=row.get('legal_form'),
                nace_code=row.get('nace_code'),
                geocoded_lat=row.get('geocoded_lat'),
                geocoded_lon=row.get('geocoded_lon'),
                quality_score=row.get('quality_score'),
            ))

    if companies_to_add:
        db.add_all(companies_to_add)
        db.commit()
        logging.info(f"Successfully added {len(companies_to_add)} new companies.")
    else:
        logging.info("No new companies to add.")

def insert_demographic_data(db: Session, df: pd.DataFrame):
    """
    Ingests demographic data into the database.
    """
    for _, row in df.iterrows():
        existing = db.query(Demographic).filter(Demographic.wilaya_code == str(row['wilaya_code'])).first()
        if existing:
            existing.population_2024 = row['population_2024']
            existing.area_km2 = row['area_km2']
            existing.density_per_km2 = row['density_per_km2']
            existing.urbanization_rate = row['urbanization_rate']
        else:
            db.add(Demographic(
                wilaya_code=str(row['wilaya_code']),
                wilaya_name=row['wilaya_name'],
                population_2024=row['population_2024'],
                area_km2=row['area_km2'],
                density_per_km2=row['density_per_km2'],
                urbanization_rate=row['urbanization_rate']
            ))
    db.commit()
    logging.info(f"Successfully ingested demographic data for {len(df)} wilayas.")

def insert_economic_indicators(db: Session, df: pd.DataFrame):
    """
    Ingests economic indicators into the database.
    """
    for _, row in df.iterrows():
        existing = db.query(EconomicIndicator).filter(EconomicIndicator.year == int(row['year'])).first()
        if existing:
            existing.gdp_growth = row['gdp_growth']
            existing.inflation = row['inflation']
            existing.oil_price_avg = row['oil_price_avg']
            existing.population_millions = row['population_millions']
            existing.unemployment_rate = row['unemployment_rate']
        else:
            db.add(EconomicIndicator(
                year=int(row['year']),
                gdp_growth=row['gdp_growth'],
                inflation=row['inflation'],
                oil_price_avg=row['oil_price_avg'],
                population_millions=row['population_millions'],
                unemployment_rate=row['unemployment_rate']
            ))
    db.commit()
    logging.info(f"Successfully ingested economic indicators for {len(df)} years.")

def insert_sectoral_data(db: Session, data: dict):
    """
    Ingests sectoral data into the database.
    """
    for sector in data.get('sectors', []):
        existing = db.query(SectoralData).filter(SectoralData.sector_name == sector['name']).first()
        if existing:
            existing.value_usd_billions = sector['value_usd_billions']
            existing.growth_rate = sector['growth_rate']
        else:
            db.add(SectoralData(
                sector_name=sector['name'],
                value_usd_billions=sector['value_usd_billions'],
                growth_rate=sector['growth_rate']
            ))
    db.commit()
    logging.info(f"Successfully ingested sectoral data for {len(data.get('sectors', []))} sectors.")
