from fastapi import APIRouter

from app.schemas import (
    InputDataOneStep,
    InputDataNoLabsOneStep,
    InputDataOneStepRecommendation,
    InputDataNoLabsOneStepRecommendation,
    RecommendDoseLabModelResponse,
    OneStepPredictionResponse,
    
)
from app.models.one_step_models import (
    predict_one_step_lab_model,
    predict_one_step_no_lab_model,
    recommend_dose_lab_model,
    recommend_dose_no_lab_model,
)

router = APIRouter()

@router.post("/inference/labs")
async def inference_labs(input_data: InputDataOneStep) -> OneStepPredictionResponse:
    return predict_one_step_lab_model(input_data)

@router.post("/inference/nolabs")
async def inference_nolabs(input_data: InputDataNoLabsOneStep) -> OneStepPredictionResponse:
    return predict_one_step_no_lab_model(input_data)


@router.post("/recommend_dose/labs")
async def recommend_dose_labs(input_data: InputDataOneStepRecommendation) -> RecommendDoseLabModelResponse:
    return recommend_dose_lab_model(input_data)

@router.post("/recommend_dose/nolabs")
async def recommend_dose_nolabs(input_data: InputDataNoLabsOneStepRecommendation) -> RecommendDoseLabModelResponse:
    return recommend_dose_no_lab_model(input_data)
