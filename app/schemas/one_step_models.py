from pydantic import BaseModel

from enum import Enum
from app.config import settings
from pathlib import Path

# Iterates the models in the saved_models folder and retrieve their names
def model_names() -> list[str]:
    models_dir = Path(settings.one_step_models_dir)
    return [model.stem for model in models_dir.glob('*.pkl')]

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

class OneStepPredictionResponse(BaseModel):
    predicted_level: float
