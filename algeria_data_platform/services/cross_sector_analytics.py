import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from ..db.models import EconomicIndicator, FinancialMarket, SectoralData, SocialIndicator
import logging

class CrossSectorAnalytics:
    def __init__(self, db: Session):
        self.db = db

    def get_market_resilience_index(self):
        """
        Calculates a market resilience index based on stock market volatility, 
        GDP growth, and sectoral diversification.
        """
        sessions = self.db.query(FinancialMarket).order_by(FinancialMarket.date).all()
        if not sessions:
            return {"error": "Insufficient financial data"}

        df_market = pd.DataFrame([{
            'date': s.date,
            'volume': s.traded_volume,
            'value': s.traded_value_dzd
        } for s in sessions])

        # Calculate volatility (std of daily returns)
        df_market['returns'] = df_market['value'].pct_change()
        volatility = df_market['returns'].std()

        # Get latest GDP growth
        latest_econ = self.db.query(EconomicIndicator).order_by(EconomicIndicator.year.desc()).first()
        gdp_growth = latest_econ.gdp_growth if latest_econ else 0

        # Resilience Index Formula (Simplified)
        # Higher GDP growth and lower volatility increase resilience
        resilience_score = (gdp_growth * 10) / (volatility + 0.1)
        
        return {
            "resilience_score": round(resilience_score, 2),
            "market_volatility": round(volatility, 4),
            "latest_gdp_growth": gdp_growth,
            "status": "Stable" if resilience_score > 50 else "Volatile"
        }

    def get_digital_readiness_report(self):
        """
        Aggregates digital and social indicators to assess national digital readiness.
        """
        indicators = self.db.query(SocialIndicator).all()
        data = {i.indicator_name: i.value for i in indicators}
        
        internet_pen = data.get('internet_penetration_rate', 0)
        mobile_pen = data.get('mobile_penetration_rate', 0)
        
        readiness_score = (internet_pen + mobile_pen) / 2
        
        return {
            "readiness_score": round(readiness_score, 2),
            "internet_penetration": internet_pen,
            "mobile_penetration": mobile_pen,
            "category": "High" if readiness_score > 70 else "Medium"
        }

    def get_infrastructure_impact_forecast(self):
        """
        Analyzes infrastructure projects and their potential economic impact.
        """
        projects = self.db.query(InfrastructureProject).all()
        total_km = sum([p.length_km for p in projects if p.length_km])
        
        return {
            "total_railway_expansion_km": total_km,
            "active_projects_count": len(projects),
            "estimated_logistics_cost_reduction": "15-20%" if total_km > 500 else "5-10%",
            "strategic_focus": "Mining and Connectivity"
        }
