from fastapi import APIRouter

from app.schemas import (
    InputDataOneStep,
    model_names,
)
from app.models.one_step_models import predict_one_step

router = APIRouter()

@router.post("/inference/{model_name}")
async def create_item(model_name: str, input_data: InputDataOneStep) -> float:
    return predict_one_step(model_name, input_data)

@router.get("/models/")
async def get_models() -> list[str]:
    return list(model_names())


