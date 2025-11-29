
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import crud
from ..schemas import salary as salary_schema
from ..db.session import get_db

router = APIRouter()

@router.get("/", response_model=List[salary_schema.SalarySchema])
def read_salaries(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Retrieve salaries.
    """
    salaries = crud.salary.get_multi(db, skip=skip, limit=limit)
    return salaries
