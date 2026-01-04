from sqlalchemy import Column, String, DateTime, func, Integer, Date, Float
from .session import Base

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
    capital_amount_dzd = Column(Integer, nullable=True)
    registration_date = Column(Date, nullable=True)
    wilaya = Column(String(2), nullable=True, index=True)
    legal_form = Column(String(10), nullable=True, index=True)
    nace_code = Column(String(10), nullable=True)
    geocoded_lat = Column(Float, nullable=True)
    geocoded_lon = Column(Float, nullable=True)
    quality_score = Column(Float, default=0.0)

class Salary(Base):
    """
    Represents a salary entry in the database.
    """
    __tablename__ = "salaries"

    id = Column(Integer, primary_key=True, index=True)
    job_title = Column(String, index=True)
    min_salary_dzd = Column(Integer)
    max_salary_dzd = Column(Integer)
    currency = Column(String)
    period = Column(String)
    source = Column(String)
    scraped_at = Column(DateTime)

class Demographic(Base):
    """
    Represents a demographic entry in the database.
    """
    __tablename__ = "demographics"

    id = Column(Integer, primary_key=True, index=True)
    wilaya_code = Column(String, unique=True, index=True)
    wilaya_name = Column(String)
    population_2024 = Column(Integer)
    area_km2 = Column(Float)
    density_per_km2 = Column(Float)
    urbanization_rate = Column(Float)
    updated_at = Column(DateTime, server_default=func.now())
