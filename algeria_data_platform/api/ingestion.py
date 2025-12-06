from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from ..database import get_db, SessionLocal
from ..services import ingestion
from ..services.ingestion import DataValidationError
from ..data_loader import load_and_clean_companies_from_csv
import pandas as pd
import io

router = APIRouter()

def process_ingestion(df: pd.DataFrame):
    """Helper function to run ingestion in the background."""
    db = SessionLocal()
    try:
        ingestion.insert_company_data(db, df)
    finally:
        db.close()

@router.post("/ingest/companies")
async def ingest_companies(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    - Ingests company data from a CSV file uploaded by the user.
    - The CSV file is read into a pandas DataFrame and then passed to the
      ingestion service to be loaded into the database in the background.
    """
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    contents = await file.read()
    buffer = io.StringIO(contents.decode("utf-8"))
    df = pd.read_csv(buffer)

    # Validate CSV headers
    required_columns = {"company_id", "legal_name", "status"}
    if not required_columns.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns. Required: {', '.join(required_columns)}"
        )

    try:
        # We run a validation check here before queueing the background task
        ingestion.validate_company_data(df)
    except DataValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={"message": "Data validation failed", "errors": e.validation_result.to_json_dict()}
        )

    background_tasks.add_task(process_ingestion, df)

    return {"message": "Company data ingestion started."}
