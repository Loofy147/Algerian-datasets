
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from algeria_data_platform.main import app
from algeria_data_platform.db.models import Salary
from datetime import datetime

client = TestClient(app)

def test_read_salaries(db_session: Session):
    salary1 = Salary(job_title="Software Engineer", min_salary_dzd=65000, max_salary_dzd=150000, currency="DZD", period="monthly", source="market_research", scraped_at=datetime.utcnow())
    salary2 = Salary(job_title="Data Analyst", min_salary_dzd=86667, max_salary_dzd=170000, currency="DZD", period="monthly", source="glassdoor", scraped_at=datetime.utcnow())
    db_session.add_all([salary1, salary2])
    db_session.commit()

    response = client.get("/api/v1/salaries/")
    assert response.status_code == 200
    assert len(response.json()) == 2
