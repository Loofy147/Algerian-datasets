from pydantic import BaseModel, Field
from typing import Optional

class EconomicIndicatorBase(BaseModel):
    """Base schema for economic indicator data."""
    indicator_name: str = Field(..., description="The name of the economic indicator.")
    year: int = Field(..., description="The year of the data.")
    value: float = Field(..., description="The value of the indicator.")
    source: Optional[str] = None

class EconomicIndicatorCreate(EconomicIndicatorBase):
    """Schema for creating a new economic indicator entry."""
    pass

class EconomicIndicatorUpdate(EconomicIndicatorBase):
    """Schema for updating an existing economic indicator entry."""
    pass

class EconomicIndicatorInDB(EconomicIndicatorBase):
    """Schema for economic indicator data in the database."""
    id: int

    class Config:
        orm_mode = True

class EconomicIndicatorSchema(EconomicIndicatorInDB):
    """Schema for returning economic indicator data from the API."""
    pass
