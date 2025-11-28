import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Adjust the import path to be relative to the project root for pytest
from algeria_data_platform.data_loader import load_and_clean_company_data

# Use a string path for patching the object where it's used
PATH_TO_PATCH = 'algeria_data_platform.data_loader.COMPANY_DATA_PATH'

@patch('pandas.read_csv')
@patch(PATH_TO_PATCH)
def test_load_and_clean_company_data_returns_dataframe_on_success(mock_path, mock_read_csv):
    """
    Tests that the function returns a pandas DataFrame when the data file exists.
    """
    mock_path.exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame({
        'company_id': [1, 2],
        'name': ['Company A', 'Company B']
    })

    result = load_and_clean_company_data()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

@patch(PATH_TO_PATCH)
def test_load_and_clean_company_data_file_not_found(mock_path):
    """
    Tests that the function returns an empty DataFrame when the file does not exist.
    """
    mock_path.exists.return_value = False

    result = load_and_clean_company_data()
    assert isinstance(result, pd.DataFrame)
    assert result.empty

@patch('pandas.read_csv')
@patch(PATH_TO_PATCH)
def test_drops_rows_with_missing_company_id(mock_path, mock_read_csv):
    """
    Tests that rows with a missing company_id are correctly dropped.
    """
    mock_path.exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame({
        'company_id': [1, np.nan, 3],
        'name': ['Company A', 'Missing ID', 'Company C']
    })

    result = load_and_clean_company_data()
    assert len(result) == 2
    assert '1' in result['company_id'].values
    assert '3' in result['company_id'].values

@patch('pandas.read_csv')
@patch(PATH_TO_PATCH)
def test_company_id_is_converted_to_string(mock_path, mock_read_csv):
    """
    Tests that the company_id column is converted from int/float to string type.
    """
    mock_path.exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame({
        'company_id': [123, 456.0],
        'name': ['Company A', 'Company B']
    })

    result = load_and_clean_company_data()
    assert result['company_id'].dtype == 'object'
    assert all(isinstance(item, str) for item in result['company_id'])
    assert result['company_id'].iloc[0] == '123'
    assert result['company_id'].iloc[1] == '456'

@patch('pandas.read_csv')
@patch(PATH_TO_PATCH)
def test_nan_values_are_replaced_with_none(mock_path, mock_read_csv):
    """
    Tests that numpy.nan values in the DataFrame are replaced with None.
    """
    mock_path.exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame({
        'company_id': [1],
        'description': [np.nan],
        'value': [100.0]
    })

    result = load_and_clean_company_data()
    records = result.to_dict(orient='records')
    assert records[0]['description'] is None
