import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from algeria_data_platform.db.models import Company
from algeria_data_platform.db.session import Base

# Use an in-memory SQLite database for testing the model
DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """
    Provides a clean database session for each test function.
    - Creates all tables in the in-memory database.
    - Yields a session to the test.
    - Drops all tables after the test is complete.
    """
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)

def test_company_model(db_session):
    """
    Tests the Company model to ensure it is correctly defined and maps to the database.
    - Creates an instance of the Company model.
    - Adds it to the session and commits to the database.
    - Queries the database to verify the company was created with the correct attributes.
    """
    company_data = {
        "company_id": "12345",
        "legal_name": "Test Company",
        "trade_name": "Test Trade Name",
        "status": "Active"
    }

    new_company = Company(**company_data)
    db_session.add(new_company)
    db_session.commit()
    db_session.refresh(new_company)

    # Verify the company was created with the correct data
    retrieved_company = db_session.query(Company).filter_by(company_id="12345").first()
    assert retrieved_company is not None
    assert retrieved_company.legal_name == "Test Company"
    assert retrieved_company.trade_name == "Test Trade Name"
    assert retrieved_company.status == "Active"
    assert retrieved_company.last_updated is not None
