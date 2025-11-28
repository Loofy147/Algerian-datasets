from typing import List
from fastapi import FastAPI
import pandas as pd
from .data_loader import load_and_clean_company_data

app = FastAPI(
    title="Algeria Data Platform API",
    description="API for accessing high-quality Algerian market data and insights.",
    version="0.1.0",
)

@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"message": "Welcome to the Algeria Data Platform API!"}

@app.get("/api/v1/companies")
def get_companies():
    """
    Retrieves a list of Algerian companies from the seed dataset.
    This endpoint serves as a proof-of-concept for the data serving layer.
    """
    company_data = load_and_clean_company_data()
    # Convert DataFrame to a list of dictionaries for JSON serialization
    return company_data.to_dict(orient="records")
