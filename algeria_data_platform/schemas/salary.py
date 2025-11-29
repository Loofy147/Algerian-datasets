
from pydantic import BaseModel
from datetime import datetime

class SalaryBase(BaseModel):
    """Base schema for salary data, containing common fields."""
    job_title: str
    min_salary_dzd: int
    max_salary_dzd: int
    currency: str
    period: str
    source: str
    scraped_at: datetime

class SalaryCreate(SalaryBase):
    """Schema for creating a new salary. Inherits from SalaryBase."""
    pass

class SalaryUpdate(SalaryBase):
    """Schema for updating an existing salary."""
    pass

class SalaryInDB(SalaryBase):
    """Schema for data retrieved from the database, including the salary ID."""
    id: int

    class Config:
        orm_mode = True

class SalarySchema(SalaryInDB):
    """The main schema for returning salary data via the API."""
    pass
