import pytest
import pandas as pd
from ..services.analytics import LassoOLSHybrid, get_economic_forecast
from ..services.ingestion import insert_demographic_data
from ..db.models import Demographic

def test_lasso_ols_hybrid():
    # Create dummy data
    X = pd.DataFrame({
        "feat1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feat2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "noise": [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.4, 0.2, 0.1, 0.3]
    })
    y = 2 * X["feat1"] + 0.5 * X["feat2"] + 5
    
    model = LassoOLSHybrid(alpha=0.01)
    model.fit(X, y)
    
    assert len(model.selected_features) > 0
    preds = model.predict(X)
    assert len(preds) == 10
    assert isinstance(preds[0], float)

def test_economic_forecast():
    data = pd.DataFrame({
        "target": [10, 12, 11, 13, 15],
        "ind1": [1, 2, 1.5, 2.5, 3],
        "ind2": [100, 105, 102, 110, 115]
    })
    
    forecast = get_economic_forecast(data, "target", forecast_steps=3)
    assert forecast["target"] == "target"
    assert len(forecast["forecast"]) == 3
    assert forecast["selected_features_count"] > 0

def test_demographic_ingestion(db_session):
    df = pd.DataFrame([{
        "wilaya_code": "16",
        "wilaya_name": "Alger",
        "population_2024": 3500000,
        "area_km2": 1190,
        "density_per_km2": 2941.18,
        "urbanization_rate": 0.95
    }])
    
    insert_demographic_data(db_session, df)
    
    demo = db_session.query(Demographic).filter(Demographic.wilaya_code == "16").first()
    assert demo is not None
    assert demo.wilaya_name == "Alger"
    assert demo.population_2024 == 3500000
