import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class LassoOLSHybrid:
    """
    A hybrid model that uses LASSO for feature selection and OLS for final prediction.
    This approach is particularly useful for economic forecasting with many potential indicators.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.lasso = Lasso(alpha=self.alpha)
        self.ols = LinearRegression()
        self.scaler = StandardScaler()
        self.selected_features = []

    def fit(self, X, y):
        """
        Fits the hybrid model.
        1. Scale features.
        2. Use LASSO to select features.
        3. Fit OLS on selected features.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.lasso.fit(X_scaled, y)
        
        # Select features with non-zero coefficients
        self.selected_features = np.where(self.lasso.coef_ != 0)[0]
        
        if len(self.selected_features) == 0:
            logger.warning("LASSO selected zero features. Falling back to all features.")
            self.selected_features = np.arange(X.shape[1])
            
        X_selected = X_scaled[:, self.selected_features]
        self.ols.fit(X_selected, y)
        logger.info(f"Model fitted. Selected {len(self.selected_features)} features.")

    def predict(self, X):
        """Predicts using the OLS model on selected features."""
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.selected_features]
        return self.ols.predict(X_selected)

def get_economic_forecast(historical_data: pd.DataFrame, target_col: str, forecast_steps: int = 5):
    """
    Generates an economic forecast using the LassoOLSHybrid model.
    """
    # Prepare data (simplified for POC)
    X = historical_data.drop(columns=[target_col])
    y = historical_data[target_col]
    
    model = LassoOLSHybrid()
    model.fit(X, y)
    
    # Generate simple future projections (simplified)
    last_row = X.iloc[-1:].copy()
    forecasts = []
    for _ in range(forecast_steps):
        pred = model.predict(last_row)[0]
        forecasts.append(float(pred))
        # Update last_row for next step (very simplified)
        last_row = last_row * 1.02 # Assume 2% growth in indicators
        
    return {
        "target": target_col,
        "forecast": forecasts,
        "selected_features_count": len(model.selected_features)
    }
