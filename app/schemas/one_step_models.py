from pydantic import BaseModel

from enum import Enum
from app.config import settings
from pathlib import Path

# Iterates the models in the saved_models folder and retrieve their names
def model_names() -> list[str]:
    models_dir = Path(settings.one_step_models_dir)
    return [model.stem for model in models_dir.glob('*.pkl')]

'''
input data schema (complete)
age                           int64
gender                        int64
race                         object
weight                        int64
height                        int64
state                        object
ast                         float64
alt                         float64
bilirubin                   float64
albumin                     float64
creatinine                  float64
sodium                      float64
potassium                   float64
hemoglobin                  float64
hematocrit                  float64
pgp_inhibit                   int64
pgp_induce                  float64
cyp3a4_inhibit                int64
cyp3a4_induce                 int64
formulation                  object
route                         int64
dose                        float64
doses_per_24_hrs              int64
level_dose_timediff           int64
treatment_days                int64
previous_dose               float64
previous_level              float64
previous_level_timediff       int64
age_group                     int64
'''

class InputDataOneStep(BaseModel):
    age: int
    gender: int
    race: str
    weight: int
    height: int
    state: int
    ast: float
    alt: float
    bilirubin: float
    albumin: float
    creatinine: float
    sodium: float
    potassium: float
    hemoglobin: float
    hematocrit: float
    pgp_inhibit: int
    pgp_induce: float
    cyp3a4_inhibit: int
    cyp3a4_induce: int
    formulation: str
    route: int
    dose: float
    doses_per_24_hrs: int
    level_dose_timediff: int
    treatment_days: int
    previous_dose: float
    previous_level: float
    previous_level_timediff: int
    age_group: int

class InputDataOneStepRecommendation(BaseModel):
    age: int
    gender: int
    race: str
    weight: int
    height: int
    state: int
    ast: float
    alt: float
    bilirubin: float
    albumin: float
    creatinine: float
    sodium: float
    potassium: float
    hemoglobin: float
    hematocrit: float
    pgp_inhibit: int
    pgp_induce: float
    cyp3a4_inhibit: int
    cyp3a4_induce: int
    formulation: str
    route: int
    dose: float
    doses_per_24_hrs: int
    level_dose_timediff: int
    treatment_days: int
    previous_dose: float
    previous_level: float
    previous_level_timediff: int
    age_group: int
    target_level: float


"""
input data schema (small)
age                       int
gender                    int
race                      str
weight                    int
height                    int
formulation               str
route                     int
dose                      float
doses_per_24_hrs          int
level_dose_timediff       int
treatment_days            int
previous_dose             float
previous_level            float
previous_level_timediff   int
age_group                 int
"""

class InputDataNoLabsOneStep(BaseModel):
    age: int
    gender: int
    race: str
    weight: int
    height: int
    formulation: str
    route: int
    dose: float
    doses_per_24_hrs: int
    level_dose_timediff: int
    treatment_days: int
    previous_dose: float
    previous_level: float
    previous_level_timediff: int
    age_group: int

class InputDataNoLabsOneStepRecommendation(BaseModel):
    age: int
    gender: int
    race: str
    weight: int
    height: int
    formulation: str
    route: int
    dose: float
    doses_per_24_hrs: int
    level_dose_timediff: int
    treatment_days: int
    previous_dose: float
    previous_level: float
    previous_level_timediff: int
    age_group: int
    target_level: float

class RecommendDoseLabModelResponse(BaseModel):
    dose_values: list[float]
    predictions: list[float]
    optimal_dose: float
    optimal_level: float