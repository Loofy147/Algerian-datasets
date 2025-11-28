from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import crud
from ..schemas import company as company_schema
from ..db.session import get_db

router = APIRouter()

@router.get("/", response_model=List[company_schema.CompanySchema])
def read_companies(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve a list of companies with pagination.
    """
    companies = crud.company.get_companies(db, skip=skip, limit=limit)
    return companies

@router.post("/", response_model=company_schema.CompanySchema)
def create_company(company: company_schema.CompanyCreate, db: Session = Depends(get_db)):
    """
    Create a new company.
    """
    db_company = crud.company.get_company(db, company_id=company.company_id)
    if db_company:
        raise HTTPException(status_code=400, detail="Company ID already registered")
    return crud.company.create_company(db=db, company=company)

@router.get("/{company_id}", response_model=company_schema.CompanySchema)
def read_company(company_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a single company by its ID.
    """
    db_company = crud.company.get_company(db, company_id=company_id)
    if db_company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return db_company

@router.put("/{company_id}", response_model=company_schema.CompanySchema)
def update_company(company_id: str, company: company_schema.CompanyUpdate, db: Session = Depends(get_db)):
    """
    Update an existing company.
    """
    db_company = crud.company.update_company(db, company_id=company_id, company=company)
    if db_company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return db_company

@router.delete("/{company_id}", response_model=company_schema.CompanySchema)
def delete_company(company_id: str, db: Session = Depends(get_db)):
    """
    Delete a company.
    """
    db_company = crud.company.delete_company(db, company_id=company_id)
    if db_company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return db_company
