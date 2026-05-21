from http import HTTPStatus

import dashboard_server
import model.predict


def test_bracket_simulation_rejects_wrong_team_count():
    payload, status = dashboard_server.bracket_simulation_payload({"format": "6", "teams": ["A", "B"]})

    assert status == HTTPStatus.BAD_REQUEST
    assert "exactly 6 teams" in payload["error"]


def test_six_team_bracket_gives_top_byes_and_normalized_winners(monkeypatch):
    formats_seen = []

    def fake_series(team_a, team_b, **kwargs):
        formats_seen.append(kwargs["series_format"])
        probability = 0.7 if team_a < team_b else 0.3
        return {"expected_win_prob": probability}

    monkeypatch.setattr(dashboard_server, "get_playground_predictor_context", lambda: object())
    monkeypatch.setattr(model.predict, "calculate_expected_series_win", fake_series)

    payload, status = dashboard_server.bracket_simulation_payload(
        {
            "format": "6",
            "series_format": "bo3",
            "grand_final_format": "bo5",
            "teams": ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"],
            "iters": 100,
        }
    )

    assert status == HTTPStatus.OK
    assert [round_info["name"] for round_info in payload["rounds"]] == [
        "Quarter-finals",
        "Semi-finals",
        "Grand final",
    ]
    assert payload["rounds"][1]["matches"][0]["left"] == ["Alpha"]
    assert payload["rounds"][1]["matches"][1]["left"] == ["Bravo"]
    assert set(payload["rounds"][1]["matches"][0]["right"]) == {"Charlie", "Foxtrot"}
    assert set(payload["rounds"][1]["matches"][1]["right"]) == {"Delta", "Echo"}
    assert payload["rounds"][2]["matches"][0]["format"] == "bo5"
    assert "bo5" in formats_seen
    assert round(sum(item["probability"] for item in payload["champion_probabilities"]), 8) == 1


def test_bracket_defaults_grand_final_to_bo5(monkeypatch):
    formats_seen = []

    def fake_series(team_a, team_b, **kwargs):
        formats_seen.append(kwargs["series_format"])
        return {"expected_win_prob": 0.5}

    monkeypatch.setattr(dashboard_server, "get_playground_predictor_context", lambda: object())
    monkeypatch.setattr(model.predict, "calculate_expected_series_win", fake_series)

    payload, status = dashboard_server.bracket_simulation_payload(
        {
            "format": "8",
            "series_format": "bo3",
            "teams": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "iters": 100,
        }
    )

    assert status == HTTPStatus.OK
    assert payload["settings"]["grand_final_format"] == "bo5"
    assert payload["rounds"][2]["matches"][0]["format"] == "bo5"
    assert "bo5" in formats_seen
