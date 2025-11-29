from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from algeria_data_platform.db.models import Company
from algeria_data_platform.schemas.company import CompanyCreate

def test_read_root(test_client: TestClient):
    """
    Tests the root endpoint ('/') to ensure it returns the correct environment.
    """
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"environment": "development"}

def test_cors_headers_are_present(test_client: TestClient):
    """
    Tests that CORS headers are present in the response when an 'Origin'
    header is included in the request.
    """
    response = test_client.get("/", headers={"Origin": "http://testserver"})
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"

def test_read_companies(db_session: Session, test_client: TestClient):
    """
    Tests the '/api/v1/companies/' endpoint for a successful data retrieval.
    """
    db_session.add(Company(company_id="1", legal_name="Test Corp A", status="Active"))
    db_session.commit()

    response = test_client.get("/api/v1/companies/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["legal_name"] == "Test Corp A"

def test_create_company(db_session: Session, test_client: TestClient):
    """
    Tests the POST '/api/v1/companies/' endpoint for creating a new company.
    """
    company_data = CompanyCreate(company_id="2", legal_name="Test Corp B", status="Active")
    response = test_client.post("/api/v1/companies/", json=company_data.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert data["legal_name"] == company_data.legal_name
    assert "last_updated" in data
