import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from algeria_data_platform.database import Base, get_db
from algeria_data_platform.main import app
from algeria_data_platform import models

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """
    Test fixture to set up a transactional, in-memory database for each test.
    - Creates a single connection for the test's duration.
    - Creates all tables.
    - Overrides the app's `get_db` dependency to use the test session.
    - Rolls back the transaction after the test to ensure isolation.
    """
    connection = engine.connect()
    transaction = connection.begin()

    # Create tables on this connection
    Base.metadata.create_all(bind=connection)

    # Create a session for the test to use
    db = TestingSessionLocal(bind=connection)

    # Override the application's dependency
    def override_get_db_for_test():
        yield db

    app.dependency_overrides[get_db] = override_get_db_for_test

    yield db

    # Teardown
    db.close()
    transaction.rollback()
    connection.close()
    # Reset the dependency override
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def test_client(db_session):
    """
    Test fixture for the FastAPI TestClient.
    Depends on db_session to ensure the database and dependency overrides are set up.
    """
    return TestClient(app)
