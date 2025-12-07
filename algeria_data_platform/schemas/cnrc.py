from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

class CNRCBase(BaseModel):
    """Base schema for CNRC data."""
    company_id: str = Field(..., description="The unique identifier for the company.")
    company_name: str = Field(..., description="The name of the company.")
    legal_form: Optional[str] = None
    capital_amount_dzd: Optional[float] = None
    registration_date: Optional[date] = None
    wilaya: Optional[str] = None

class CNRCCreate(CNRCBase):
    """Schema for creating a new CNRC entry."""
    pass

class CNRCUpdate(CNRCBase):
    """Schema for updating an existing CNRC entry."""
    pass

class CNRCInDB(CNRCBase):
    """
    Schema for CNRC data in the database.
    Note: company_id serves as the primary key.
    """
    class Config:
        orm_mode = True

class CNRCSchema(CNRCInDB):
    """Schema for returning CNRC data from the API."""
    pass
