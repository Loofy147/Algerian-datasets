# Algeria Data Platform

The Algeria Data Platform is a FastAPI-based web service designed to provide high-quality, cleaned, and validated data related to the Algerian market. It serves as a centralized API for accessing insights and datasets for various economic, demographic, and business analyses.

## Project Structure

-   `algeria_data_platform/`: The core Python package containing the FastAPI application.
    -   `main.py`: The main entry point for the FastAPI application.
    -   `data_loader.py`: Handles loading and cleaning of the seed data.
    -   `data/`: Contains the raw data files.
    -   `tests/`: Contains the unit tests for the application.
-   `operations/`: Contains documentation related to engineering and operational best practices.
-   `requirements.txt`: The main application dependencies.
-   `requirements-dev.txt`: The development and testing dependencies.

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
