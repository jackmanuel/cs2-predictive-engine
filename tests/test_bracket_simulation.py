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


def test_sixteen_team_swiss_simulation(monkeypatch):
    def fake_series(team_a, team_b, **kwargs):
        probability = 0.6 if team_a < team_b else 0.4
        return {"expected_win_prob": probability}

    monkeypatch.setattr(dashboard_server, "get_playground_predictor_context", lambda: object())
    monkeypatch.setattr(model.predict, "calculate_expected_series_win", fake_series)

    teams = [f"Team_{i}" for i in range(1, 17)]
    payload, status = dashboard_server.bracket_simulation_payload(
        {
            "format": "16",
            "series_format": "bo3",
            "teams": teams,
            "iters": 100,
        }
    )

    assert status == HTTPStatus.OK
    assert payload["format"] == 16
    assert len(payload["rounds"]) == 5
    assert len(payload["champion_probabilities"]) == 16

    # Assert Round 1 matches follow the 1 vs 9, 2 vs 10, etc. seeding rule
    round_1_matches = payload["rounds"][0]["matches"]
    assert len(round_1_matches) == 8
    for i in range(8):
        match = round_1_matches[i]
        assert match["left"] == [f"Team_{i + 1}"]
        assert match["right"] == [f"Team_{i + 9}"]

    for item in payload["champion_probabilities"]:
        assert "team" in item
        assert "probability" in item
        assert "records" in item
        assert isinstance(item["records"], dict)


def test_sixteen_team_swiss_pickem_optimization(monkeypatch):
    def fake_series(team_a, team_b, **kwargs):
        # Even probability for balanced Monte Carlo simulation distribution
        return {"expected_win_prob": 0.5}

    monkeypatch.setattr(dashboard_server, "get_playground_predictor_context", lambda: object())
    monkeypatch.setattr(model.predict, "calculate_expected_series_win", fake_series)

    teams = [f"Team_{i}" for i in range(1, 17)]
    payload, status = dashboard_server.bracket_simulation_payload(
        {
            "format": "16",
            "series_format": "bo3",
            "teams": teams,
            "iters": 10,
        }
    )

    assert status == HTTPStatus.OK
    assert "pickem_optimization" in payload
    
    pickem = payload["pickem_optimization"]
    assert "success_probability" in pickem
    assert isinstance(pickem["success_probability"], float)
    assert 0.0 <= pickem["success_probability"] <= 1.0

    picks_30 = pickem["picks_3_0"]
    picks_03 = pickem["picks_0_3"]
    picks_qual = pickem["picks_qual"]

    # Official Pick 'ems counts: exactly 2 for 3-0, 2 for 0-3, and 6 for non-undefeated qualifiers
    assert len(picks_30) == 2
    assert len(picks_03) == 2
    assert len(picks_qual) == 6

    # Verify disjointness - selections must be completely unique to avoid duplicate allocations
    all_picks = picks_30 + picks_03 + picks_qual
    assert len(set(all_picks)) == 10
    
    # Ensure they are valid competing teams
    for team in all_picks:
        assert team in teams


