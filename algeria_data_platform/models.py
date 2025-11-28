from sqlalchemy import Column, String, DateTime, func
from .database import Base

class Company(Base):
    """
    Represents a company entry in the database.
    - Maps to the 'companies' table.
    - Includes fields for company identification, legal names, and status.
    - The 'last_updated' field is automatically managed by the database.
    """
    __tablename__ = "companies"

    company_id = Column(String, primary_key=True, index=True)
    legal_name = Column(String)
    trade_name = Column(String, nullable=True)
    status = Column(String)
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())
