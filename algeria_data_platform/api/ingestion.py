from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from ..db.session import get_db, SessionLocal
from .generic_ingestion import insert_data, validate_data, INGESTION_REGISTRY, DataValidationError
import pandas as pd
import io
import logging

router = APIRouter()

def process_ingestion(data_type: str, df: pd.DataFrame):
    """Helper function to run ingestion in the background."""
    logging.info(f"process_ingestion started for data_type: {data_type}")
    db = SessionLocal()
    try:
        logging.info("Calling insert_data")
        insert_data(db, data_type, df)
        logging.info("insert_data finished")
    except Exception as e:
        logging.error(f"Error during ingestion: {e}")
    finally:
        db.close()
        logging.info("process_ingestion finished")

@router.post("/ingest/{data_type}")
async def ingest_data_endpoint(
    data_type: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    - Ingests data from a CSV file uploaded by the user.
    - The CSV file is read into a pandas DataFrame and then passed to the
      ingestion service to be loaded into the database in the background.
    """
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    if data_type not in INGESTION_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Invalid data type: {data_type}")

    contents = await file.read()
    buffer = io.StringIO(contents.decode("utf-8"))

    config = INGESTION_REGISTRY[data_type]
    pk_column = config.get("pk")
    dtype_mapping = {pk_column: str} if pk_column else None
    df = pd.read_csv(buffer, dtype=dtype_mapping)

    # Validate CSV headers
    required_columns = set(config["column_mapping"].keys())

    if not required_columns.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns for {data_type}. Required: {', '.join(required_columns)}"
        )

    try:
        # We run a validation check here before queueing the background task
        validate_data(data_type, df)
    except DataValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={"message": "Data validation failed", "errors": e.validation_result.to_json_dict()}
        )

    background_tasks.add_task(process_ingestion, data_type, df)

    return {"message": f"{data_type.capitalize()} data ingestion started."}
