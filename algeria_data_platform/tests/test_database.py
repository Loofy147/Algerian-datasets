import pytest
from sqlalchemy.orm import Session
from algeria_data_platform.db.session import get_db, SessionLocal, engine, Base

def test_get_db():
    """
    Tests the get_db dependency to ensure it yields a valid database session.
    - Verifies that the object returned is an instance of Session.
    - Ensures the session is properly closed after use.
    """
    db_generator = get_db()
    db_session = next(db_generator)

    assert isinstance(db_session, Session)

    # Ensure the session is closed after the generator is exhausted
    try:
        next(db_generator)
    except StopIteration:
        pass

    assert db_session._transaction is None
