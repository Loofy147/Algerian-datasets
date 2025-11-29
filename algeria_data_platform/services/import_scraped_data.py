
import pandas as pd
from sqlalchemy.orm import Session
from ..db.session import SessionLocal
from ..crud import salary as crud_salary
from ..schemas import salary as salary_schema
from datetime import datetime

def import_salaries_from_csv(filepath: str, db: Session = None):
    """
    Imports salary data from a CSV file into the database.
    """
    if db is None:
        db = SessionLocal()
    df = pd.read_csv(filepath)
    for _, row in df.iterrows():
        salary_in = salary_schema.SalaryCreate(
            job_title=row["job_title"],
            min_salary_dzd=row["min_salary_dzd"],
            max_salary_dzd=row["max_salary_dzd"],
            currency=row["currency"],
            period=row["period"],
            source=row["source"],
            scraped_at=datetime.fromisoformat(row["scraped_at"])
        )
        crud_salary.create(db, obj_in=salary_in)
    db.close()

if __name__ == "__main__":
    import_salaries_from_csv("scraped_data/salaries_algeria_2025.csv")
