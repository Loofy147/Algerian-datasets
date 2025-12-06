import pandas as pd
from faker import Faker
import random
from datetime import datetime

# Initialize Faker for synthetic data generation
fake = Faker('fr_FR')
Faker.seed(0)
random.seed(0)

# Define job titles and their salary ranges
JOB_TITLES = {
    "Développeur de logiciels": (80000, 150000),
    "Ingénieur DevOps": (100000, 180000),
    "Chef de projet": (120000, 200000),
    "Data Scientist": (110000, 190000),
    "Responsable marketing": (90000, 160000),
    "Comptable": (60000, 110000),
}

def generate_salary_data(num_records: int) -> pd.DataFrame:
    """
    Generates a DataFrame with synthetic salary data.
    """
    data = []
    for _ in range(num_records):
        job_title = random.choice(list(JOB_TITLES.keys()))
        min_salary, max_salary = JOB_TITLES[job_title]

        data.append({
            "job_title": job_title,
            "min_salary_dzd": min_salary,
            "max_salary_dzd": max_salary,
            "currency": "DZD",
            "period": "Monthly",
            "source": fake.company(),
            "scraped_at": "2025-12-06 12:00:00",
        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    NUM_RECORDS = 50
    df = generate_salary_data(NUM_RECORDS)

    # Save the generated data to a CSV file
    from pathlib import Path
    DATA_DIR = Path(__file__).parent
    SALARY_DATA_PATH = DATA_DIR / "salary_sample_data.csv"
    df.to_csv(SALARY_DATA_PATH, index=False)

    print(f"Successfully generated {NUM_RECORDS} salary records.")
    print(df.head())
