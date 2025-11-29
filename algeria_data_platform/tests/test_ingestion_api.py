from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from algeria_data_platform.main import app
from algeria_data_platform.db.models import Company
from algeria_data_platform.api import ingestion as ingestion_api
from fastapi import BackgroundTasks
import io

client = TestClient(app)

# Create a class to run background tasks synchronously for testing
class SyncBackgroundTasks(BackgroundTasks):
    def add_task(self, func, *args, **kwargs) -> None:
        func(*args, **kwargs)

def test_ingest_companies(db_session: Session, monkeypatch):
    """
    - Tests the /api/v1/ingest/companies endpoint.
    - Uploads a sample CSV file and verifies that the data is correctly
      ingested into the database.
    - Checks that duplicate entries are not created on subsequent uploads.
    """
    # Override BackgroundTasks to run tasks synchronously
    app.dependency_overrides[BackgroundTasks] = lambda: SyncBackgroundTasks()

    # Monkeypatch SessionLocal to use the test db_session and prevent it from being closed
    monkeypatch.setattr(ingestion_api, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)

    # Sample CSV data
    csv_data = """company_id,legal_name,trade_name,status
1,Test Company 1,TC1,Active
2,Test Company 2,TC2,Inactive
"""

    # First upload
    response = client.post(
        "/api/v1/ingest/companies",
        files={"file": ("test.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Company data ingestion started."}

    # Verify data in the database (should be present now)
    companies = db_session.query(Company).all()
    assert len(companies) == 2

    # Second upload of the same data (should not create duplicates)
    response = client.post(
        "/api/v1/ingest/companies",
        files={"file": ("test.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    )
    assert response.status_code == 200

    # Verify that no new companies were added
    companies = db_session.query(Company).all()
    assert len(companies) == 2

    # Clean up the dependency override
    app.dependency_overrides.clear()


def test_ingest_companies_malformed_csv(db_session: Session):
    """
    - Tests the /api/v1/ingest/companies endpoint with a malformed CSV.
    - Uploads a CSV file with missing required columns and verifies that a
      400 Bad Request error is returned.
    """
    # Sample CSV data with missing 'legal_name' column
    csv_data = """company_id,trade_name,status
1,TC1,Active
"""

    response = client.post(
        "/api/v1/ingest/companies",
        files={"file": ("test.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    )
    assert response.status_code == 400
    assert "Missing required columns" in response.json()["detail"]

def test_ingest_companies_data_quality_validation(db_session: Session):
    """
    - Tests the /api/v1/ingest/companies endpoint with data that violates
      the Great Expectations suite.
    - Uploads a CSV file with a duplicate company ID and verifies that a
      400 Bad Request error is returned with validation details.
    """
    # Sample CSV data with a duplicate company ID
    csv_data = """company_id,legal_name,trade_name,status
1,Test Company 1,TC1,Active
1,Test Company 2,TC2,Inactive
"""

    response = client.post(
        "/api/v1/ingest/companies",
        files={"file": ("test.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    )
    assert response.status_code == 400
    response_json = response.json()
    assert response_json["detail"]["message"] == "Data validation failed"
    assert "errors" in response_json["detail"]
    assert response_json["detail"]["errors"]["results"][2]["expectation_config"]["expectation_type"] == "expect_column_values_to_be_unique"
