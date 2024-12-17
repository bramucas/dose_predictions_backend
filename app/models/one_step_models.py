import joblib
from pandas import DataFrame

from fastapi import HTTPException
from app.schemas.one_step_models import (
    InputDataOneStep,
    InputDataNoLabsOneStep,
)
from app.config import settings

lab_model   = None
nolab_model = None

def predict_one_step(model, input_data) -> float:
    # Convert input_data (Pydantic model) to a pandas DataFrame with the expected feature columns
    input_data_dict = input_data.model_dump()  # Get dictionary from the Pydantic model
    input_data_df = DataFrame([input_data_dict])  # Create DataFrame for a single sample

    return model.predict(input_data_df)[0]

def predict_one_step_lab_model(input_data: InputDataOneStep) -> float:
    """
    Makes a prediction using the default no_lab model, that is the model without the lab features.
    It's more accurate than the 'no_labs' one.
    """
    # Load pre-trained model if not already loaded
    lab_model = joblib.load(f'{settings.one_step_model_labs_path}')
    
    # TEST TODO
    # test_data = joblib.load(f'{settings.one_step_models_dir}/test_data.joblib')

    prediction = predict_one_step(lab_model, input_data)

    return prediction

def predict_one_step_no_lab_model(input_data: InputDataNoLabsOneStep) -> float:
    """
    Makes a prediction using the default no_lab model, that is the model without the lab features.
    """
    nolab_model = joblib.load(f'{settings.one_step_model_nolabs_path}')
    
    # TEST TODO
    # test_data = joblib.load(f'{settings.one_step_models_dir}/test_data.joblib')

    prediction = predict_one_step(nolab_model, input_data)

    return prediction
