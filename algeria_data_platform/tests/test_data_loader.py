import pandas as pd
from sqlalchemy.orm import Session
from algeria_data_platform.data_loader import get_all_companies_as_df
from algeria_data_platform.models import Company

def test_get_all_companies_as_df_returns_dataframe(db_session: Session):
    """
    Tests that the function returns a pandas DataFrame when the database has data.
    """
    # Add some test data
    db_session.add(Company(company_id='123', legal_name='Test Company', status='Active'))
    db_session.commit()

    result = get_all_companies_as_df(db_session)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.iloc[0]['legal_name'] == 'Test Company'

def test_get_all_companies_as_df_empty_database(db_session: Session):
    """
    Tests that the function returns an empty DataFrame when the database is empty.
    """
    result = get_all_companies_as_df(db_session)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
