# Algeria Data Platform

The Algeria Data Platform is a production-grade FastAPI-based web service designed to provide high-quality, cleaned, and validated data related to the Algerian market. It serves as a centralized API for accessing insights, datasets, and economic forecasts.

## Key Features

-   **Multi-Source Ingestion**: Automated pipelines for company, salary, and demographic data.
-   **Data Quality Assurance**: Integrated with **Great Expectations** for rigorous validation.
-   **Advanced Analytics**: Implementation of the **LASSO-OLS Hybrid** model for economic forecasting.
-   **Algeria-Specific Logic**: Built-in support for wilaya codes, legal forms, and local regulations (Law 18-07).
-   **Operational Excellence**: Comprehensive runbooks, logging, and error handling.

## Project Structure

-   `algeria_data_platform/`: The core Python package.
    -   `api/`: FastAPI routers (Companies, Salaries, Analytics, Ingestion).
    -   `services/`: Business logic and analytics models.
    -   `db/`: Database models and session management.
    -   `data/`: Raw and processed datasets.
    -   `core/`: Configuration and logging.
-   `gx/`: Great Expectations configuration and expectation suites.
-   `operations/`: Runbooks, checklists, and compliance documentation.
-   `migrations/`: Alembic database migrations.

## Getting Started

### Prerequisites

-   Python 3.10+
-   `pip` and `venv`

### Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

### Running the Application

To run the FastAPI server locally, use the following command from the root of the project:

```bash
uvicorn algeria_data_platform.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### Running the Tests

To run the unit tests, use the following command from the root of the project:

```bash
PYTHONPATH=. pytest -v
```
