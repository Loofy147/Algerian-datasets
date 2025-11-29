
from sqlalchemy.orm import Session
from typing import Optional
from .base import CRUDBase
from ..db import models
from ..schemas import company as company_schema


class CRUDCompany(CRUDBase[models.Company, company_schema.CompanyCreate, company_schema.CompanyUpdate]):
    def get(self, db: Session, id: str) -> Optional[models.Company]:
        return db.query(self.model).filter(self.model.company_id == id).first()

company = CRUDCompany(models.Company)
