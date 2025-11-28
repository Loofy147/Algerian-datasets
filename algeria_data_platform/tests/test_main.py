from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from algeria_data_platform.models import Company
from algeria_data_platform.main import app

# The test_client fixture is now defined in conftest.py and does not need to be imported here

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

def test_get_companies_endpoint_success(db_session: Session, test_client: TestClient):
    """
    Tests the '/api/v1/companies' endpoint for a successful data retrieval
    from the test database.
    """
    # Add some test data to the in-memory database
    db_session.add_all([
        Company(company_id='1', legal_name='Test Corp A', status='Active'),
        Company(company_id='2', legal_name='Test Corp B', status='Inactive')
    ])
    db_session.commit()

    response = test_client.get("/api/v1/companies")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2
    assert data[0]['legal_name'] == 'Test Corp A'
    assert data[1]['status'] == 'Inactive'

def test_get_companies_endpoint_empty_database(test_client: TestClient):
    """
    Tests the '/api/v1/companies' endpoint when the database is empty.
    """
    response = test_client.get("/api/v1/companies")
    assert response.status_code == 200
    assert response.json() == []
