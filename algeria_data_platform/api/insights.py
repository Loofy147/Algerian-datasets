from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..db.session import get_db
from ..services.cross_sector_analytics import CrossSectorAnalytics

router = APIRouter(prefix="/insights", tags=["Insights"])

@router.get("/market-resilience")
def get_market_resilience(db: Session = Depends(get_db)):
    analytics = CrossSectorAnalytics(db)
    return analytics.get_market_resilience_index()

@router.get("/digital-readiness")
def get_digital_readiness(db: Session = Depends(get_db)):
    analytics = CrossSectorAnalytics(db)
    return analytics.get_digital_readiness_report()

@router.get("/infrastructure-impact")
def get_infrastructure_impact(db: Session = Depends(get_db)):
    analytics = CrossSectorAnalytics(db)
    return analytics.get_infrastructure_impact_forecast()
