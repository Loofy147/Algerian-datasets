from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from algeria_data_platform.main import app
from algeria_data_platform.db.models import Company

client = TestClient(app)

def test_read_company(db_session: Session):
    company = Company(company_id="1", legal_name="Test Company", status="Active")
    db_session.add(company)
    db_session.commit()
    response = client.get("/api/v1/companies/1")
    assert response.status_code == 200
    assert response.json()["legal_name"] == "Test Company"

def test_update_company(db_session: Session):
    company = Company(company_id="1", legal_name="Test Company", status="Active")
    db_session.add(company)
    db_session.commit()
    response = client.put("/api/v1/companies/1", json={"legal_name": "Updated Company", "status": "Inactive"})
    assert response.status_code == 200
    assert response.json()["legal_name"] == "Updated Company"
    assert response.json()["status"] == "Inactive"

def test_delete_company(db_session: Session):
    company = Company(company_id="1", legal_name="Test Company", status="Active")
    db_session.add(company)
    db_session.commit()
    response = client.delete("/api/v1/companies/1")
    assert response.status_code == 200
    assert response.json()["company_id"] == "1"
    response = client.get("/api/v1/companies/1")
    assert response.status_code == 404
