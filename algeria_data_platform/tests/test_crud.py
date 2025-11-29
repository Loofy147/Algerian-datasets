
from sqlalchemy.orm import Session
from algeria_data_platform.crud import company as crud_company
from algeria_data_platform.schemas import company as company_schema
from algeria_data_platform.db import models

def test_create_company(db_session: Session):
    company_in = company_schema.CompanyCreate(company_id="1", legal_name="Test Company", status="Active")
    company = crud_company.create(db_session, obj_in=company_in)
    assert company.company_id == "1"
    assert company.legal_name == "Test Company"

def test_get_company(db_session: Session):
    company_in = company_schema.CompanyCreate(company_id="1", legal_name="Test Company", status="Active")
    crud_company.create(db_session, obj_in=company_in)
    company = crud_company.get(db_session, id="1")
    assert company.company_id == "1"
    assert company.legal_name == "Test Company"

def test_update_company(db_session: Session):
    company_in = company_schema.CompanyCreate(company_id="1", legal_name="Test Company", status="Active")
    company = crud_company.create(db_session, obj_in=company_in)
    company_update = company_schema.CompanyUpdate(legal_name="Updated Company", status="Inactive")
    company2 = crud_company.update(db_session, db_obj=company, obj_in=company_update)
    assert company2.legal_name == "Updated Company"
    assert company2.status == "Inactive"

def test_delete_company(db_session: Session):
    company_in = company_schema.CompanyCreate(company_id="1", legal_name="Test Company", status="Active")
    crud_company.create(db_session, obj_in=company_in)
    company = crud_company.remove(db_session, id="1")
    company2 = crud_company.get(db_session, id="1")
    assert company2 is None
