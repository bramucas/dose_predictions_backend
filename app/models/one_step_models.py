import joblib

from fastapi import HTTPException
from app.schemas.one_step_models import InputDataOneStep
from app.config import settings

from numpy import array
from pandas import DataFrame

models = dict()

def predict_one_step(model_name:str, input_data: InputDataOneStep) -> float:
    # Load pre-trained model if not already loaded
    if model_name not in models:
        try:
            models[model_name] = joblib.load(f'{settings.one_step_models_dir}/{model_name}.pkl')
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Model not found")
    # TEST TODO
    # test_data = joblib.load(f'{settings.one_step_models_dir}/test_data.joblib')

    # Convert input_data (Pydantic model) to a pandas DataFrame with the expected feature columns
    input_data_dict = input_data.model_dump()  # Get dictionary from the Pydantic model
    input_data_df = DataFrame([input_data_dict])  # Create DataFrame for a single sample

    return models[model_name].predict(input_data_df)[0]
