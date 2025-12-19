from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root_not_found():
    response = client.get("/")
    assert response.status_code in (200, 404)

def test_transcribe_no_file():
    response = client.post("/transcribe")
    assert response.status_code == 422
