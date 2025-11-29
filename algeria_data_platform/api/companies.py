"""
Enhanced API for Algeria Data Platform
Adds Algeria-specific features: wilaya filtering, quality metrics, Arabic support
"""

from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
import logging

from .. import crud
from ..schemas import company as company_schema
from ..db.session import get_db
from ..db.models import Company

logger = logging.getLogger(__name__)

router = APIRouter()

# Wilaya codes mapping (48 wilayas in Algeria)
WILAYA_CODES = {
    "01": "Adrar", "02": "Chlef", "03": "Laghouat", "04": "Oum El Bouaghi",
    "05": "Batna", "06": "Béjaïa", "07": "Biskra", "08": "Béchar",
    "09": "Blida", "10": "Bouira", "11": "Tamanrasset", "12": "Tébessa",
    "13": "Tlemcen", "14": "Tiaret", "15": "Tizi Ouzou", "16": "Alger",
    "17": "Djelfa", "18": "Jijel", "19": "Sétif", "20": "Saïda",
    "21": "Skikda", "22": "Sidi Bel Abbès", "23": "Annaba", "24": "Guelma",
    "25": "Constantine", "26": "Médéa", "27": "Mostaganem", "28": "M'Sila",
    "29": "Mascara", "30": "Ouargla", "31": "Oran", "32": "El Bayadh",
    "33": "Illizi", "34": "Bordj Bou Arréridj", "35": "Boumerdès", "36": "El Tarf",
    "37": "Tindouf", "38": "Tissemsilt", "39": "El Oued", "40": "Khenchela",
    "41": "Souk Ahras", "42": "Tipaza", "43": "Mila", "44": "Aïn Defla",
    "45": "Naâma", "46": "Aïn Témouchent", "47": "Ghardaïa", "48": "Relizane"
}


@router.get("/", response_model=List[company_schema.CompanySchema])
def read_companies(
    skip: int = 0,
    limit: int = Query(default=100, le=1000),  # Max 1000 per page
    wilaya: Optional[str] = Query(None, description="Filter by wilaya code (01-48)"),
    legal_form: Optional[str] = Query(None, description="Filter by legal form (SARL, SPA, etc.)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    min_capital: Optional[int] = Query(None, description="Minimum capital in DZD"),
    db: Session = Depends(get_db)
):
    """
    Retrieve companies with advanced filtering

    **Algeria-specific filters:**
    - `wilaya`: Filter by administrative region (01-48)
    - `legal_form`: SARL, SPA, EURL, SNC, etc.
    - `min_capital`: Minimum registered capital in Algerian Dinars
    """
    query = db.query(Company)

    # Apply filters
    if wilaya:
        if wilaya not in WILAYA_CODES:
            raise HTTPException(status_code=400, detail=f"Invalid wilaya code. Must be 01-48.")
        # Assuming wilaya is stored in a separate column
        query = query.filter(Company.wilaya == wilaya)

    if legal_form:
        valid_forms = ['SARL', 'SPA', 'EURL', 'SNC', 'SCS']
        if legal_form not in valid_forms:
            raise HTTPException(status_code=400, detail=f"Invalid legal form. Must be one of: {', '.join(valid_forms)}")
        query = query.filter(Company.legal_form == legal_form)

    if status:
        query = query.filter(Company.status == status)

    if min_capital:
        query = query.filter(Company.capital_amount_dzd >= min_capital)

    # Execute query with pagination
    companies = query.offset(skip).limit(limit).all()

    logger.info(f"Retrieved {len(companies)} companies (wilaya={wilaya}, legal_form={legal_form})")
    return companies


@router.get("/stats", response_model=Dict)
def get_platform_statistics(db: Session = Depends(get_db)):
    """
    Get overall platform statistics

    Returns:
    - Total companies
    - Breakdown by wilaya
    - Breakdown by legal form
    - Data quality score
    - Last update timestamp
    """
    total_companies = db.query(func.count(Company.company_id)).scalar()

    # Breakdown by legal form (if column exists)
    # legal_form_breakdown = db.query(
    #     Company.legal_form,
    #     func.count(Company.company_id)
    # ).group_by(Company.legal_form).all()

    # Data quality metrics (simplified)
    null_legal_name = db.query(func.count(Company.company_id)).filter(
        Company.legal_name.is_(None)
    ).scalar()

    quality_score = ((total_companies - null_legal_name) / total_companies * 100) if total_companies > 0 else 0

    last_update = db.query(func.max(Company.last_updated)).scalar()

    return {
        "total_companies": total_companies,
        "data_quality_score": round(quality_score, 2),
        "last_updated": last_update.isoformat() if last_update else None,
        "coverage": {
            "wilayas_covered": 48,  # Placeholder
            "legal_forms": ["SARL", "SPA", "EURL", "SNC"],
        },
        "platform_status": "operational",
        "api_version": "v1.0.0"
    }


@router.get("/wilayas", response_model=List[Dict])
def get_wilaya_statistics(db: Session = Depends(get_db)):
    """
    Get company statistics by wilaya (administrative region)

    Returns breakdown of companies per wilaya with economic indicators
    """

    results = db.query(
        Company.wilaya,
        func.count(Company.company_id).label("companies_count"),
        func.avg(Company.capital_amount_dzd).label("avg_capital_dzd")
    ).group_by(Company.wilaya).all()

    wilaya_stats = []
    for row in results:
        wilaya_stats.append({
            "code": row.wilaya,
            "name": WILAYA_CODES.get(row.wilaya, "Unknown"),
            "companies_count": row.companies_count,
            "avg_capital_dzd": row.avg_capital_dzd,
            "most_common_sector": "N/A" # Placeholder
        })

    return wilaya_stats


@router.get("/quality-report", response_model=Dict)
def get_data_quality_report(db: Session = Depends(get_db)):
    """
    Real-time data quality report

    Implements the "Red Team" philosophy by exposing quality metrics publicly
    """
    total_records = db.query(func.count(Company.company_id)).scalar()

    # Calculate quality dimensions
    quality_checks = {
        "completeness": {
            "company_id": 100.0,  # Primary key always complete
            "legal_name": _calculate_completeness(db, Company.legal_name, total_records),
            "trade_name": _calculate_completeness(db, Company.trade_name, total_records),
            "status": _calculate_completeness(db, Company.status, total_records),
        },
        "timeliness": {
            "last_updated_24h": _count_recent_updates(db, hours=24),
            "last_updated_7d": _count_recent_updates(db, hours=7*24),
            "avg_data_age_hours": _calculate_avg_age(db),
        },
        "uniqueness": {
            "duplicate_ids": _count_duplicates(db),
            "duplicate_rate": 0.0,  # Calculate from above
        }
    }

    # Overall quality score (weighted average)
    completeness_score = sum(quality_checks["completeness"].values()) / len(quality_checks["completeness"])
    timeliness_score = 100.0 if quality_checks["timeliness"]["avg_data_age_hours"] < 24 else 80.0
    uniqueness_score = 100.0 - (quality_checks["uniqueness"]["duplicate_rate"] * 100)

    overall_score = (completeness_score * 0.4 + timeliness_score * 0.3 + uniqueness_score * 0.3)

    return {
        "overall_quality_score": round(overall_score, 2),
        "quality_dimensions": quality_checks,
        "assessment_timestamp": datetime.utcnow().isoformat(),
        "meets_sla": overall_score >= 95.0,
        "recommendations": _generate_quality_recommendations(quality_checks)
    }


@router.get("/search", response_model=List[company_schema.CompanySchema])
def search_companies(
    q: str = Query(..., min_length=2, description="Search query (company name, trade name)"),
    limit: int = Query(default=50, le=100),
    db: Session = Depends(get_db)
):
    """
    Full-text search across company names

    Supports Arabic, French, and transliterated text
    """
    # Simple LIKE search (would use full-text search in production)
    search_pattern = f"%{q}%"

    companies = db.query(Company).filter(
        (Company.legal_name.ilike(search_pattern)) |
        (Company.trade_name.ilike(search_pattern))
    ).limit(limit).all()

    logger.info(f"Search query='{q}' returned {len(companies)} results")
    return companies


@router.post("/", response_model=company_schema.CompanySchema)
def create_company(obj_in: company_schema.CompanyCreate, db: Session = Depends(get_db)):
    return crud.company.create(db=db, obj_in=obj_in)

@router.get("/{company_id}", response_model=company_schema.CompanySchema)
def read_company(company_id: str, db: Session = Depends(get_db)):
    db_company = crud.company.get(db, id=company_id)
    if db_company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return db_company

@router.put("/{company_id}", response_model=company_schema.CompanySchema)
def update_company(company_id: str, company: company_schema.CompanyUpdate, db: Session = Depends(get_db)):
    db_company = crud.company.get(db, id=company_id)
    if db_company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return crud.company.update(db=db, db_obj=db_company, obj_in=company)

@router.delete("/{company_id}", response_model=company_schema.CompanySchema)
def delete_company(company_id: str, db: Session = Depends(get_db)):
    db_company = crud.company.get(db, id=company_id)
    if db_company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return crud.company.remove(db=db, id=company_id)

# ============= Helper Functions =============

def _calculate_completeness(db: Session, column, total: int) -> float:
    """Calculate percentage of non-null values"""
    if total == 0:
        return 0.0
    non_null = db.query(func.count(column)).filter(column.isnot(None)).scalar()
    return round((non_null / total) * 100, 2)


def _count_recent_updates(db: Session, hours: int) -> int:
    """Count records updated in last N hours"""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    return db.query(func.count(Company.company_id)).filter(
        Company.last_updated >= cutoff
    ).scalar()


def _calculate_avg_age(db: Session) -> float:
    """Calculate average age of data in hours"""
    now = datetime.utcnow()

    # Use a dialect-specific function for converting datetime to timestamp
    if db.bind.dialect.name == "sqlite":
        avg_last_updated_timestamp = db.query(func.avg(func.strftime('%s', Company.last_updated))).scalar()
    else:
        avg_last_updated_timestamp = db.query(func.avg(func.extract('epoch', Company.last_updated))).scalar()

    if avg_last_updated_timestamp:
        avg_last_updated = datetime.utcfromtimestamp(avg_last_updated_timestamp)
        age = (now - avg_last_updated).total_seconds() / 3600
        return round(age, 2)
    return 0.0


def _count_duplicates(db: Session) -> int:
    """Count duplicate company IDs"""
    duplicates = db.query(Company.company_id, func.count(Company.company_id)).group_by(
        Company.company_id
    ).having(func.count(Company.company_id) > 1).count()
    return duplicates


def _generate_quality_recommendations(checks: Dict) -> List[str]:
    """Generate actionable recommendations based on quality metrics"""
    recommendations = []

    # Check completeness
    for field, completeness in checks["completeness"].items():
        if completeness < 90:
            recommendations.append(f"Improve {field} completeness (currently {completeness}%)")

    # Check timeliness
    if checks["timeliness"]["avg_data_age_hours"] > 48:
        recommendations.append("Data freshness below target (>48h old on average)")

    # Check uniqueness
    if checks["uniqueness"]["duplicate_ids"] > 0:
        recommendations.append(f"Remove {checks['uniqueness']['duplicate_ids']} duplicate records")

    if not recommendations:
        recommendations.append("All quality metrics meet SLA targets")

    return recommendations
