import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging_config import setup_logging
from .api import companies, ingestion, salaries, analytics, insights
from fastapi.responses import JSONResponse

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred.", "detail": str(exc)},
    )

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Include the companies router
app.include_router(companies.router, prefix="/api/v1/companies", tags=["companies"])
app.include_router(salaries.router, prefix="/api/v1/salaries", tags=["salaries"])
app.include_router(ingestion.router, prefix="/api/v1", tags=["ingestion"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(insights.router, prefix="/api/v1/insights", tags=["insights"])
