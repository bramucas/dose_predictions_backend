from fastapi import APIRouter

from app.schemas import (
    InputDataOneStep,
    InputDataNoLabsOneStep,
)
from app.models.one_step_models import predict_one_step_lab_model, predict_one_step_no_lab_model

router = APIRouter()

@router.post("/inference/labs")
async def create_item(input_data: InputDataOneStep) -> float:
    return predict_one_step_lab_model(input_data)

@router.post("/inference/nolabs")
async def create_item(input_data: InputDataNoLabsOneStep) -> float:
    return predict_one_step_no_lab_model(input_data)


