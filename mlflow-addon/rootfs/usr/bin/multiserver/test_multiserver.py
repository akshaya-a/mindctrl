from fastapi.testclient import TestClient
import pandas as pd

from .main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert list(response.json().keys()) == ["Tracking Store", "Model Registry"]


def test_score_model():
    df = pd.DataFrame({"animal": ["cats", "dogs"]})
    payload = {"dataframe_split": df.to_dict(orient="split")}
    response = client.post(
        "/deployed-models/chat/versions/1/invocations",
        json=payload,
    )
    assert response.status_code == 200
    jokes = response.json()
    assert len(jokes) == 2
    assert "cat" in jokes[0]
    assert "dog" in jokes[1]
