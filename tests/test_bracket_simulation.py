from collections import Counter
from http import HTTPStatus

import dashboard_server
import model.predict
from config import MC_ITERATIONS, SWISS_STAGE_ITERATIONS, SWISS_VETO_ITERATIONS
from model import veto_sim


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
    veto_iterations_seen = set()

    def fake_series(team_a, team_b, **kwargs):
        formats_seen.append(kwargs["series_format"])
        veto_iterations_seen.add(kwargs["iters"])
        return {"expected_win_prob": 0.5}

    monkeypatch.setattr(dashboard_server, "get_playground_predictor_context", lambda: object())
    monkeypatch.setattr(model.predict, "calculate_expected_series_win", fake_series)

    payload, status = dashboard_server.bracket_simulation_payload(
        {
            "format": "8",
            "series_format": "bo3",
            "teams": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "swiss_iters": 300,
        }
    )

    assert status == HTTPStatus.OK
    assert payload["settings"]["grand_final_format"] == "bo5"
    assert payload["rounds"][2]["matches"][0]["format"] == "bo5"
    assert "bo5" in formats_seen
    assert payload["settings"]["swiss_iterations"] is None
    assert veto_iterations_seen == {MC_ITERATIONS}


def test_sixteen_team_swiss_simulation(monkeypatch):
    veto_iterations_seen = set()

    def fake_series(team_a, team_b, **kwargs):
        veto_iterations_seen.add(kwargs["iters"])
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
            "veto_iters": "200",
            "swiss_iters": "100",
        }
    )

    assert status == HTTPStatus.OK
    assert payload["format"] == 16
    assert payload["settings"]["veto_iterations"] == 200
    assert payload["settings"]["swiss_iterations"] == 100
    assert payload["settings"]["iterations"] == 100
    assert veto_iterations_seen == {200}
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
        assert sum(item["records"].values()) == 100


def test_swiss_iteration_defaults_are_independent(monkeypatch):
    veto_iterations_seen = set()

    def fake_series(team_a, team_b, **kwargs):
        veto_iterations_seen.add(kwargs["iters"])
        return {"expected_win_prob": 0.5}

    def fake_swiss(teams, series_format, threshold, iters, lookup, *, bo3_only):
        assert iters == SWISS_STAGE_ITERATIONS
        lookup(teams[0], teams[8], "bo1")
        return [], [], None

    monkeypatch.setattr(dashboard_server, "get_playground_predictor_context", lambda: object())
    monkeypatch.setattr(dashboard_server, "simulate_swiss_stage", fake_swiss)
    monkeypatch.setattr(model.predict, "calculate_expected_series_win", fake_series)

    payload, status = dashboard_server.bracket_simulation_payload(
        {"format": "16", "teams": [f"Team_{i}" for i in range(1, 17)]}
    )

    assert status == HTTPStatus.OK
    assert payload["settings"]["veto_iterations"] == SWISS_VETO_ITERATIONS
    assert payload["settings"]["swiss_iterations"] == SWISS_STAGE_ITERATIONS
    assert veto_iterations_seen == {SWISS_VETO_ITERATIONS}


def test_swiss_reuses_veto_stats_and_map_scores_per_request(monkeypatch):
    teams = [f"TEAM {i}" for i in range(1, 17)]
    history_reads = []
    team_stats_calls = []
    map_scores = Counter()
    ctx = object()

    def load_data():
        history = object()
        history_reads.append(history)
        return history

    def get_team_stats(team_id, history):
        assert history is history_reads[-1]
        team_stats_calls.append(team_id)
        return team_id

    def run_simulations(stats_a, stats_b, *, iters, series_format, starts_veto):
        sequences = {
            "bo1": "Mirage",
            "bo3": "Mirage,Nuke,Ancient",
            "bo5": "Mirage,Inferno,Dust2,Nuke,Ancient",
        }
        return {sequences[series_format]: iters}, {}

    def get_win_probabilities(predictor_ctx, team_a, team_b, maps, *, picker_override):
        assert predictor_ctx is ctx
        for map_name in maps:
            map_scores[team_a, team_b, map_name, picker_override] += 1
        return [0.5] * len(maps)

    def simulate_swiss_stage(teams, series_format, threshold, iters, lookup, *, bo3_only):
        for team_a, team_b in zip(teams[:8], teams[8:]):
            lookup(team_a, team_b, "bo1")
        lookup(teams[0], teams[8], "bo3")
        lookup(teams[0], teams[8], "bo5")
        lookup(teams[8], teams[0], "bo3")
        return [], [], None

    monkeypatch.setattr(dashboard_server, "get_playground_predictor_context", lambda: ctx)
    monkeypatch.setattr(dashboard_server, "simulate_swiss_stage", simulate_swiss_stage)
    monkeypatch.setattr(veto_sim, "load_data", load_data)
    monkeypatch.setattr(veto_sim, "get_team_stats", get_team_stats)
    monkeypatch.setattr(veto_sim, "run_simulations", run_simulations)
    monkeypatch.setattr(model.predict, "get_win_probabilities", get_win_probabilities)

    opening_pairs = list(zip(teams[:8], teams[8:]))
    first_a, first_b = opening_pairs[0]
    expected_scores = {(a, b, "Mirage", "neutral") for a, b in opening_pairs}
    expected_scores.update(
        (first_a, first_b, map_name, picker)
        for map_name in ("Mirage", "Nuke", "Inferno", "Dust2")
        for picker in ("a", "b")
    )
    expected_scores.update(
        (first_b, first_a, map_name, picker)
        for map_name in ("Mirage", "Nuke")
        for picker in ("a", "b")
    )
    expected_scores.update({(first_a, first_b, "Ancient", "neutral"), (first_b, first_a, "Ancient", "neutral")})

    request = {"format": "16", "teams": [team.lower() for team in teams], "iters": 100}
    for request_count in (1, 2):
        payload, status = dashboard_server.bracket_simulation_payload(request)

        assert status == HTTPStatus.OK
        assert payload["matchup_count"] == 11
        assert len(history_reads) == request_count
        assert Counter(team_stats_calls) == {team: request_count for team in teams}
        assert map_scores == {key: request_count for key in expected_scores}


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
    assert payload["settings"]["veto_iterations"] == 100
    assert payload["settings"]["swiss_iterations"] == 100
    
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


def test_sixteen_team_swiss_simulation_bo3_only(monkeypatch):
    formats_seen = set()

    def fake_series(team_a, team_b, **kwargs):
        formats_seen.add(kwargs.get("series_format"))
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
            "bo3_only": True,
        }
    )

    assert status == HTTPStatus.OK
    assert payload["format"] == 16
    assert payload["settings"]["bo3_only"] is True
    # In BO3-only Swiss, all matches should be evaluated/simulated as BO3, and only BO3 should be in formats_seen
    assert formats_seen == {"bo3"}
    
    # Also verify that every match returned has "BO3" format
    for round_info in payload["rounds"]:
        for match in round_info["matches"]:
            assert match["format"] == "BO3"
