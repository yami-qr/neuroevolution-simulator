"""The dashboard and assets share the backend's origin."""

from fastapi.testclient import TestClient

from neuroevolution.server.api import create_app


def test_dashboard_and_assets_are_served_by_fastapi(database):
    with TestClient(create_app(database)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Evolution lab" in page.text
        assert "/assets/app.js" in page.text
        assert client.get("/assets/styles.css").status_code == 200
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/api/v1/defaults").status_code == 200
