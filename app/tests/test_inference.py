from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_inference_one_step_prediction():
    response = client.post("/inference/model_a", json={
        "feature_1": 1.2,
        "feature_2": 3,
        "feature_3": "some_text"
    })
    assert response.status_code == 200
    assert "result" in response.json()
