import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from .data_loader import load_and_clean_company_data
from .core.config import settings

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
def get_companies():
    """
    Retrieves a list of Algerian companies from the seed dataset.
    This endpoint serves as a proof-of-concept for the data serving layer.
    """
    company_data = load_and_clean_company_data()
    # Convert DataFrame to a list of dictionaries for JSON serialization
    return company_data.to_dict(orient="records")
