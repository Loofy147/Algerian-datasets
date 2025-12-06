from sqlalchemy import Column, String, DateTime, func, Integer, Float
from .database import Base

class Salary(Base):
    """
    Represents a salary entry in the database.
    - Maps to the 'salaries' table.
    - Includes fields for job title, salary range, and data source.
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
    - Maps to the 'demographics' table.
    - Includes fields for population, births, and deaths by year and wilaya.
    """
    __tablename__ = "demographics"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, index=True)
    wilaya = Column(String, index=True)
    population = Column(Integer)
    births = Column(Integer)
    deaths = Column(Integer)

class EconomicIndicator(Base):
    """
    Represents an economic indicator entry in the database.
    - Maps to the 'economic_indicators' table.
    - Includes fields for the indicator name, year, and value.
    """
    __tablename__ = "economic_indicators"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, index=True)
    indicator_name = Column(String, index=True)
    value = Column(Float)

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
