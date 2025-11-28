from sqlalchemy.orm import Session
from algeria_data_platform import crud
from algeria_data_platform.schemas.company import CompanyCreate, CompanyUpdate
from algeria_data_platform.db.models import Company

def test_create_company(db_session: Session):
    company_in = CompanyCreate(company_id="1", legal_name="Test Company", status="Active")
    company = crud.company.create_company(db_session, company=company_in)
    assert company.legal_name == company_in.legal_name
    assert company.company_id == company_in.company_id

def test_get_company(db_session: Session):
    company = Company(company_id="1", legal_name="Test Company", status="Active")
    db_session.add(company)
    db_session.commit()
    retrieved_company = crud.company.get_company(db_session, company_id="1")
    assert retrieved_company.legal_name == "Test Company"

def test_get_companies(db_session: Session):
    company1 = Company(company_id="1", legal_name="Test Company 1", status="Active")
    company2 = Company(company_id="2", legal_name="Test Company 2", status="Active")
    db_session.add_all([company1, company2])
    db_session.commit()
    companies = crud.company.get_companies(db_session)
    assert len(companies) == 2

def test_update_company(db_session: Session):
    company = Company(company_id="1", legal_name="Test Company", status="Active")
    db_session.add(company)
    db_session.commit()
    company_update = CompanyUpdate(legal_name="Updated Company", status="Inactive")
    updated_company = crud.company.update_company(db_session, company_id="1", company=company_update)
    assert updated_company.legal_name == "Updated Company"
    assert updated_company.status == "Inactive"

def test_delete_company(db_session: Session):
    company = Company(company_id="1", legal_name="Test Company", status="Active")
    db_session.add(company)
    db_session.commit()
    deleted_company = crud.company.delete_company(db_session, company_id="1")
    assert deleted_company.company_id == "1"
    retrieved_company = crud.company.get_company(db_session, company_id="1")
    assert retrieved_company is None
