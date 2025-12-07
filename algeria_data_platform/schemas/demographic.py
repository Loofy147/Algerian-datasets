from pydantic import BaseModel, Field
from typing import Optional

class DemographicBase(BaseModel):
    """Base schema for demographic data."""
    wilaya: str = Field(..., description="The wilaya (province) of the demographic data.")
    year: int = Field(..., description="The year of the demographic data.")
    population: Optional[int] = None
    source: Optional[str] = None

class DemographicCreate(DemographicBase):
    """Schema for creating a new demographic entry."""
    pass

class DemographicUpdate(DemographicBase):
    """Schema for updating an existing demographic entry."""
    pass

class DemographicInDB(DemographicBase):
    """Schema for demographic data in the database."""
    id: int

    class Config:
        orm_mode = True

class DemographicSchema(DemographicInDB):
    """Schema for returning demographic data from the API."""
    pass
