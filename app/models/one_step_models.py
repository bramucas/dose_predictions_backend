import joblib
from numpy import linspace, argmin
from pandas import DataFrame


from app.schemas.one_step_models import (
    InputDataOneStep,
    InputDataNoLabsOneStep,
    InputDataOneStepRecommendation,
    InputDataNoLabsOneStepRecommendation,
    RecommendDoseLabModelResponse,
    OneStepPredictionResponse,
)
from app.config import settings

def predict_one_step(model, input_data) -> float:
    '''
    Uses the given model to make a prediction of the tacro level.
    '''
    # Convert input_data (Pydantic model) to a pandas DataFrame with the expected feature columns
    input_data_dict = input_data.model_dump()  # Get dictionary from the Pydantic model
    input_data_df = DataFrame([input_data_dict])  # Create DataFrame for a single sample

    return model.predict(input_data_df)[0]

def get_recommended_dose(model, input_data) -> float:
    '''
    Given a model and a target level, it predicts the level for different doses to find the optimal dose.
    It returns the optimal dose and the corresponding predictions.
    '''
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

def predict_one_step_lab_model(input_data: InputDataOneStep) -> OneStepPredictionResponse:
    """
    Makes a prediction using the default no_lab model, that is the model without the lab features.
    It's more accurate than the 'no_labs' one.
    """
    # Load pre-trained model if not already loaded
    lab_model = joblib.load(f'{settings.one_step_model_labs_path}')

    prediction = predict_one_step(lab_model, input_data)

    response = OneStepPredictionResponse(predicted_level=float(round(prediction.item(), 2)))

    return response

def predict_one_step_no_lab_model(input_data: InputDataNoLabsOneStep) -> OneStepPredictionResponse:
    """
    Makes a prediction using the default no_lab model, that is the model without the lab features.
    """
    nolab_model = joblib.load(f'{settings.one_step_model_nolabs_path}')

    prediction = predict_one_step(nolab_model, input_data)

    response = OneStepPredictionResponse(predicted_level=float(round(prediction.item(), 2)))

    return response


def recommend_dose_lab_model(input_data: InputDataOneStepRecommendation) -> RecommendDoseLabModelResponse:
    '''
    Returns the recommended dose for a given input_data, as well as the corresponding predictions
    used to compute the optimal dose.
    '''
    # Load pre-trained model if not already loaded
    lab_model = joblib.load(f'{settings.one_step_model_labs_path}')

    dose_values, predictions, optimal_dose, optimal_level = get_recommended_dose(lab_model, input_data)

    response = RecommendDoseLabModelResponse(
        dose_values=[round(float(x), 2) for x in dose_values],
        predictions=[round(float(x), 2) for x in predictions],
        optimal_dose=round(float(optimal_dose), 2),
        optimal_level=round(float(optimal_level), 2),
    )

    return response

def recommend_dose_no_lab_model(input_data: InputDataNoLabsOneStepRecommendation) -> RecommendDoseLabModelResponse:
    '''
    Returns the recommended dose for a given input_data, as well as the corresponding predictions
    used to compute the optimal dose. It uses the *no_labs model*.
    '''
    # Load pre-trained model if not already loaded
    nolab_model = joblib.load(f'{settings.one_step_model_nolabs_path}')

    dose_values, predictions, optimal_dose, optimal_level = get_recommended_dose(nolab_model, input_data)

    response = RecommendDoseLabModelResponse(
        dose_values=[round(float(x), 2) for x in dose_values],
        predictions=[round(float(x), 2) for x in predictions],
        optimal_dose=round(float(optimal_dose), 2),
        optimal_level=round(float(optimal_level), 2),
    )

    return response
