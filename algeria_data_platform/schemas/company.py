from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date

class CompanyBase(BaseModel):
    """Base schema for company data, containing common fields."""
    legal_name: str
    trade_name: Optional[str] = None
    status: str
    capital_amount_dzd: Optional[int] = None
    registration_date: Optional[date] = None
    wilaya: Optional[str] = None
    legal_form: Optional[str] = None
    nace_code: Optional[str] = None
    geocoded_lat: Optional[float] = None
    geocoded_lon: Optional[float] = None
    quality_score: Optional[float] = None

class CompanyCreate(CompanyBase):
    """Schema for creating a new company. Inherits from CompanyBase."""
    company_id: str

class CompanyUpdate(CompanyBase):
    """Schema for updating an existing company."""
    pass

class CompanyInDB(CompanyBase):
    """Schema for data retrieved from the database, including the company ID."""
    company_id: str
    last_updated: datetime

    class Config:
        orm_mode = True

class CompanySchema(CompanyInDB):
    """The main schema for returning company data via the API."""
    pass
