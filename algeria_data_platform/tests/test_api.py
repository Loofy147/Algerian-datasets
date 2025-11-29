from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from algeria_data_platform.main import app
from algeria_data_platform.db.models import Company

client = TestClient(app)

def test_read_companies(db_session: Session):
    company1 = Company(company_id="1", legal_name="Test Company 1", status="Active", wilaya="16", legal_form="SARL", capital_amount_dzd=100000)
    company2 = Company(company_id="2", legal_name="Test Company 2", status="Active", wilaya="31", legal_form="SPA", capital_amount_dzd=200000)
    db_session.add_all([company1, company2])
    db_session.commit()

    response = client.get("/api/v1/companies/")
    assert response.status_code == 200
    assert len(response.json()) == 2

    response = client.get("/api/v1/companies/?wilaya=16")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["legal_name"] == "Test Company 1"

    response = client.get("/api/v1/companies/?legal_form=SPA")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["legal_name"] == "Test Company 2"

    response = client.get("/api/v1/companies/?min_capital=150000")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["legal_name"] == "Test Company 2"

def test_get_wilayas_statistics(db_session: Session):
    company1 = Company(company_id="1", legal_name="Test Company 1", status="Active", wilaya="16", capital_amount_dzd=100000)
    company2 = Company(company_id="2", legal_name="Test Company 2", status="Active", wilaya="16", capital_amount_dzd=200000)
    company3 = Company(company_id="3", legal_name="Test Company 3", status="Active", wilaya="31", capital_amount_dzd=300000)
    db_session.add_all([company1, company2, company3])
    db_session.commit()

    response = client.get("/api/v1/companies/wilayas")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    alger_stats = next((item for item in data if item["code"] == "16"), None)
    assert alger_stats["companies_count"] == 2
    assert alger_stats["avg_capital_dzd"] == 150000

def test_get_platform_statistics(db_session: Session):
    company1 = Company(company_id="1", legal_name="Test Company 1", status="Active")
    company2 = Company(company_id="2", legal_name="Test Company 2", status="Active")
    db_session.add_all([company1, company2])
    db_session.commit()

    response = client.get("/api/v1/companies/stats")
    assert response.status_code == 200
    assert response.json()["total_companies"] == 2

def test_get_data_quality_report(db_session: Session):
    company1 = Company(company_id="1", legal_name="Test Company 1", status="Active")
    company2 = Company(company_id="2", trade_name="Trade Name 2", status="Active")
    db_session.add_all([company1, company2])
    db_session.commit()

    response = client.get("/api/v1/companies/quality-report")
    assert response.status_code == 200
    assert "overall_quality_score" in response.json()

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
