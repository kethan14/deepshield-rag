from fastapi.testclient import TestClient

from deepshield_api.main import app

client = TestClient(app)


def test_health_endpoint_returns_service_status() -> None:
    """The health endpoint should report the API identity and version."""

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "deepshield-rag-api",
        "version": "0.1.0",
    }
