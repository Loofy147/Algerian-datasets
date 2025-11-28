import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from algeria_data_platform.database import Base
from algeria_data_platform.models import Company
from algeria_data_platform.seed_db import seed_database
from unittest.mock import patch
import logging

# Configure logging to capture output during tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use an in-memory SQLite database for testing the seeding logic
DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def test_db():
    """
    Provides a clean in-memory database for each test function.
    - Creates all tables.
    - Yields the database engine.
    - Drops all tables after the test.
    """
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def mock_csv_data(tmp_path):
    """
    Creates a mock CSV file with sample company data for testing.
    - The CSV contains a mix of new companies and duplicates.
    - Returns the path to the temporary CSV file.
    """
    data = {
        'company_id': ['1', '2', '3', '2'],  # '2' is a duplicate
        'legal_name': ['Company A', 'Company B', 'Company C', 'Company B'],
        'trade_name': ['Trade A', 'Trade B', 'Trade C', 'Trade B'],
        'status': ['Active', 'Active', 'Inactive', 'Active']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_companies.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_seed_database(test_db, mock_csv_data, caplog):
    """
    Tests the seed_database function to ensure it correctly populates the database.
    - Mocks the CSV data path to use the temporary test file.
    - Verifies that new companies are added and duplicates are skipped.
    - Checks that the final database state is as expected.
    """
    caplog.set_level(logging.INFO)
    with patch('algeria_data_platform.seed_db.COMPANY_DATA_PATH', mock_csv_data):
        with patch('algeria_data_platform.seed_db.engine', engine):
            # Run the seeding process
            seed_database()

            # Verify the database state
            session = TestingSessionLocal()
            companies = session.query(Company).all()

            assert len(companies) == 3  # Should skip the duplicate

            # Check that the correct companies were added
            company_ids = {c.company_id for c in companies}
            assert company_ids == {'1', '2', '3'}

            session.close()

            # Check logging output for success message
            assert "Successfully added 3 new companies" in caplog.text
