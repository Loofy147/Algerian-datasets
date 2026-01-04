# Runbook: Data Ingestion Pipeline

**Version**: 1.0
**Owner**: Data Engineering Team
**Last Updated**: 2025-11-29

## Overview
This runbook describes the process for ingesting new datasets into the Algeria Data Platform, ensuring data quality and compliance with Law 18-07.

## Prerequisites
- Access to the production/staging environment.
- Python 3.10+ installed.
- Database credentials.

## Step-by-Step Process

### 1. Data Preparation
- Ensure the raw data is in CSV format.
- Place the file in `algeria_data_platform/data/raw/`.
- Verify that the file follows the naming convention: `{source}_{type}_data.csv`.

### 2. Validation (Great Expectations)
- Run the validation suite before ingestion:
  ```bash
  pytest algeria_data_platform/tests/test_data_loader.py
  ```
- If validation fails, check the `gx/uncommitted/data_docs` for detailed reports.

### 3. Ingestion Execution
- Use the `seed_db.py` script for initial seeding or the API endpoint `/api/v1/ingest` for incremental updates.
  ```bash
  python -m algeria_data_platform.seed_db
  ```

### 4. Post-Ingestion Verification
- Check the API stats endpoint:
  ```bash
  curl http://localhost:8000/api/v1/companies/stats
  ```
- Verify the `data_quality_score` is above 95%.

## Troubleshooting
- **Error: DataValidationError**: The incoming data violates the expectation suite. Check for null values in primary keys or invalid wilaya codes.
- **Error: ConnectionError**: Database is unreachable. Check the `DATABASE_URL` in `.env`.

## Compliance Check
- Ensure no PII (Personally Identifiable Information) is ingested without proper anonymization.
- Verify that the data source is authorized under Law 18-07.
