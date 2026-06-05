from api.index import app


def test_predict_endpoint_returns_pitwit_payload():
    client = app.test_client()

    response = client.post(
        "/api/predict",
        json={
            "car_id": "ALB",
            "lap": 42,
            "lap_distance_m": 2410,
            "speed_kph": 178,
            "throttle": 0.71,
            "brake": 0,
            "tire_age_laps": 29,
            "track_temp_c": 42,
            "air_temp_c": 27,
            "compound": "MEDIUM",
            "track_status": "VSC",
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["frame"]["telemetry"]["car_id"] == "ALB"
    assert 0 <= body["frame"]["inference"]["pit_risk"] <= 1
    assert body["model_source"] == "demo_model/pitwit_demo.onnx"
    assert body["summary"]["frames"] == 4
    assert "<svg" in body["svg"]
