import joblib
from numpy import linspace, argmin
from pandas import DataFrame


from fastapi import HTTPException
from app.schemas.one_step_models import (
    InputDataOneStep,
    InputDataNoLabsOneStep,
    InputDataOneStepRecommendation,
    InputDataNoLabsOneStepRecommendation
)
from app.config import settings

lab_model   = None
nolab_model = None

def predict_one_step(model, input_data) -> float:
    # Convert input_data (Pydantic model) to a pandas DataFrame with the expected feature columns
    input_data_dict = input_data.model_dump()  # Get dictionary from the Pydantic model
    input_data_df = DataFrame([input_data_dict])  # Create DataFrame for a single sample

    return model.predict(input_data_df)[0]

def get_recommended_dose(model, input_data) -> float:
    # Convert input_data (Pydantic model) to a pandas DataFrame with the expected feature columns
    input_data_dict = input_data.model_dump()  # Get dictionary from the Pydantic model
    target_level = input_data_dict['target_level']
    del input_data_dict['target_level']
    input_data_df = DataFrame([input_data_dict])  # Create DataFrame for a single sample

    dose_range = (0.5, 10)  # Fixed?

    dose_values = linspace(start=dose_range[0], stop=dose_range[1], num=22)
    predictions = []
    errors = []
    for dose in dose_values:
        row_df_copy = input_data_df.copy()
        row_df_copy['dose'] = dose
        predicted_level = model.predict(row_df_copy)[0]
        predictions.append(predicted_level)
        error = abs(predicted_level - target_level)
        errors.append(error)

    min_error_index = argmin(errors)
    optimal_dose = dose_values[min_error_index]
    optimal_level = predictions[min_error_index]

    return dose_values, predictions, optimal_dose, optimal_level

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


def recommend_dose_lab_model(input_data: InputDataOneStepRecommendation) -> float:
    # Load pre-trained model if not already loaded
    lab_model = joblib.load(f'{settings.one_step_model_labs_path}')

    dose_values, predictions, optimal_dose, optimal_level = get_recommended_dose(lab_model, input_data)

    return optimal_dose

def recommend_dose_no_lab_model(input_data: InputDataNoLabsOneStepRecommendation) -> float:
    # Load pre-trained model if not already loaded
    nolab_model = joblib.load(f'{settings.one_step_model_nolabs_path}')

    dose_values, predictions, optimal_dose, optimal_level = get_recommended_dose(nolab_model, input_data)

    return optimal_dose