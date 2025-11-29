
from .base import CRUDBase
from ..db import models
from ..schemas import salary as salary_schema

class CRUDSalary(CRUDBase[models.Salary, salary_schema.SalaryCreate, salary_schema.SalaryUpdate]):
    pass

salary = CRUDSalary(models.Salary)
