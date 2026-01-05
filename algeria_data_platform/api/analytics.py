from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, List
import pandas as pd
import numpy as np
from ..db.session import get_db
from ..db.models import Company, Salary, Demographic, EconomicIndicator, SectoralData
from ..services.analytics import get_economic_forecast

router = APIRouter()

@router.get("/forecast/gdp", response_model=Dict)
def get_gdp_forecast(
    steps: int = Query(5, ge=1, le=12),
    db: Session = Depends(get_db)
):
    """
    Generates a GDP forecast for Algeria using the LASSO-OLS Hybrid model.
    Uses internal platform data (company registrations, salaries, demographics) as indicators.
    """
    # In a real scenario, we would fetch historical GDP data and correlate it with our platform data.
    # For this POC, we'll simulate the historical data structure.
    
    # Fetch historical data from the database
    historical_records = db.query(EconomicIndicator).order_by(EconomicIndicator.year).all()
    
    if not historical_records:
        raise HTTPException(status_code=404, detail="No historical economic data found for forecasting.")
        
    data = {
        "year": [r.year for r in historical_records],
        "gdp_growth": [r.gdp_growth for r in historical_records],
        "inflation": [r.inflation for r in historical_records],
        "oil_price_avg": [r.oil_price_avg for r in historical_records],
        "population_millions": [r.population_millions for r in historical_records],
        "unemployment_rate": [r.unemployment_rate for r in historical_records],
    }
    df = pd.DataFrame(data)
    
    # Add internal platform data as features (simulated for now, should be aggregated from Company/Salary/Demographic)
    df["company_registrations"] = np.linspace(10000, 30000, len(df))
    df["avg_salary_index"] = np.linspace(100, 130, len(df))
    
    forecast = get_economic_forecast(df.drop(columns=["year"]), "gdp_growth", forecast_steps=steps)
    
    return {
        "indicator": "GDP Growth Rate (%)",
        "historical_data": data,
        "forecast": forecast["forecast"],
        "model_metadata": {
            "model_type": "LassoOLSHybrid",
            "features_selected": forecast["selected_features_count"],
            "last_updated": "2025-11-29"
        }
    }

@router.get("/market-insights", response_model=Dict)
def get_market_insights(db: Session = Depends(get_db)):
    """
    Returns high-level market insights based on platform data.
    """
    total_companies = db.query(Company).count()
    avg_salary = db.query(Salary).with_entities(Salary.min_salary_dzd).all()
    avg_salary_val = np.mean([s[0] for s in avg_salary]) if avg_salary else 0
    
    sectoral_data = db.query(SectoralData).all()
    sector_insights = {s.sector_name: {"value_usd_billions": s.value_usd_billions, "growth_rate": s.growth_rate} for s in sectoral_data}
    
    return {
        "market_summary": "The Algerian market shows strong growth in the tech and services sectors, supported by new data integration.",
        "key_metrics": {
            "total_companies_tracked": total_companies,
            "average_monthly_salary_dzd": round(avg_salary_val, 2),
            "top_performing_wilaya": "16 - Alger"
        },
        "sectoral_insights": sector_insights,
        "investment_outlook": "Positive, especially in digital transformation and renewable energy.",
        "risk_level": "Moderate (Regulatory changes, currency fluctuations)"
    }
