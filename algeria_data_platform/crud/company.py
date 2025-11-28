from sqlalchemy.orm import Session
from ..db import models
from ..schemas import company as company_schema

def get_company(db: Session, company_id: str):
    """
    Retrieves a single company from the database by its ID.
    """
    return db.query(models.Company).filter(models.Company.company_id == company_id).first()

def get_companies(db: Session, skip: int = 0, limit: int = 100):
    """
    Retrieves a list of companies from the database with optional pagination.
    """
    return db.query(models.Company).offset(skip).limit(limit).all()

def create_company(db: Session, company: company_schema.CompanyCreate):
    """
    Creates a new company in the database.
    """
    db_company = models.Company(**company.dict())
    db.add(db_company)
    db.commit()
    db.refresh(db_company)
    return db_company
