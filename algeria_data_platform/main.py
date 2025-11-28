import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from sqlalchemy.orm import Session
from fastapi import Depends
from .data_loader import get_all_companies_as_df
from .core.config import settings
from .database import get_db

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
)

# --- Middleware ---
# 1. CORS Middleware for handling cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# 2. Logging Middleware to capture request details
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs incoming request headers for better operational visibility."""
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.debug(f"Headers: {request.headers}")
    response = await call_next(request)
    return response

@app.get("/")
def read_root():
    """Returns the current environment name."""
    return {"environment": settings.ENV}

@app.get("/api/v1/companies")
def get_companies(db: Session = Depends(get_db)):
    """
    Retrieves a list of Algerian companies from the database.
    """
    company_data = get_all_companies_as_df(db)
    return company_data.to_dict(orient="records")
