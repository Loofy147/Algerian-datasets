
import pytest
from algeria_data_platform.services.red_team_tests import RedTeamDataQuality
from algeria_data_platform.services.ingestion import validate_company_data
from algeria_data_platform.data_loader import load_and_clean_companies_from_csv
import pandas as pd

@pytest.fixture
def red_team_tester():
    return RedTeamDataQuality()

from algeria_data_platform.data_loader import COMPANY_DATA_PATH

@pytest.fixture
def clean_df():
    # Using the actual data loader to ensure the test is realistic
    return load_and_clean_companies_from_csv(COMPANY_DATA_PATH)

def test_red_team_resilience(red_team_tester, clean_df):
    """
    Runs the full suite of red team adversarial tests and generates a report.
    The test will fail if the resilience score is below a certain threshold.
    """
    # Run adversarial tests
    results = red_team_tester.run_adversarial_tests(clean_df, validate_company_data)

    # Generate report
    report = red_team_tester.generate_resilience_report(results)
    print(report)

    # Assert that the resilience score is above a certain threshold
    total_tests = 0
    passed_tests = 0
    for attack_type in results:
        for attack_name in results[attack_type]:
            total_tests += 1
            if results[attack_type][attack_name]['passed']:
                passed_tests += 1

    resilience_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    assert resilience_score >= 80
