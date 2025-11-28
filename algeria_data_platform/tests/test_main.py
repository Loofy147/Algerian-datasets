from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd

# Import the FastAPI app instance from your main application file
from algeria_data_platform.main import app

# Create a TestClient instance
client = TestClient(app)

def test_read_root():
    """
    Tests the root endpoint ('/') to ensure it returns the correct environment.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"environment": "development"}  # Default value from config

def test_cors_headers_are_present():
    """
    Tests that CORS headers are present in the response when an 'Origin'
    header is included in the request.
    """
    # CORS headers are only added if the request includes an 'Origin' header
    response = client.get("/", headers={"Origin": "http://testserver"})
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"

@patch('algeria_data_platform.main.load_and_clean_company_data')
def test_get_companies_endpoint_success(mock_load_data):
    """
    Tests the '/api/v1/companies' endpoint for a successful data retrieval.
    - Mocks the data loader to return a sample DataFrame.
    - Verifies the status code and the structure of the JSON response.
    """
    # Configure the mock to return a sample DataFrame
    mock_load_data.return_value = pd.DataFrame({
        'company_id': ['123', '456'],
        'name': ['Test Corp A', 'Test Corp B'],
        'description': ['A test company', None]
    })

    response = client.get("/api/v1/companies")
    assert response.status_code == 200

    # Verify the response body
    expected_data = [
        {'company_id': '123', 'name': 'Test Corp A', 'description': 'A test company'},
        {'company_id': '456', 'name': 'Test Corp B', 'description': None}
    ]
    assert response.json() == expected_data

@patch('algeria_data_platform.main.load_and_clean_company_data')
def test_get_companies_endpoint_empty_dataframe(mock_load_data):
    """
    Tests the '/api/v1/companies' endpoint when the data loader returns an empty DataFrame.
    - Ensures the API returns an empty list without errors.
    """
    # Configure the mock to return an empty DataFrame
    mock_load_data.return_value = pd.DataFrame()

    response = client.get("/api/v1/companies")
    assert response.status_code == 200
    assert response.json() == []
