
from sqlalchemy.orm import Session
from algeria_data_platform.services.import_scraped_data import import_salaries_from_csv
from algeria_data_platform.db.models import Salary

def test_import_salaries_from_csv(db_session: Session):
    # Create a dummy CSV file
    csv_content = """job_title,min_salary_dzd,max_salary_dzd,currency,period,source,scraped_at
Software Engineer,65000,150000,DZD,monthly,market_research,2025-11-29T13:51:43.654948
Data Analyst,86667,170000,DZD,monthly,glassdoor,2025-11-29T13:51:43.654948
"""
    with open("test_salaries.csv", "w") as f:
        f.write(csv_content)

    try:
        # Import the data
        import_salaries_from_csv("test_salaries.csv", db=db_session)

        # Check that the data was imported correctly
        salaries = db_session.query(Salary).all()
        assert len(salaries) == 2
        assert salaries[0].job_title == "Software Engineer"
        assert salaries[1].job_title == "Data Analyst"
    finally:
        # Clean up the dummy CSV file
        import os
        os.remove("test_salaries.csv")
