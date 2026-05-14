from datetime import datetime, timezone

from evaluation.veto_backtest import (
    DEFAULT_ERAS,
    FeatureRow,
    ModelSpec,
    VetoHistory,
    normalize_name,
    parse_veto_match,
    predict_probabilities,
)


def test_parse_veto_match_tracks_bo1_team_ban_slots():
    record = {
        "url": "https://www.hltv.org/matches/1/a-vs-b",
        "date": "2026-05-01",
        "format": "bo1",
        "team1": "Team A",
        "team2": "Team B",
        "hltv_vetoes": [
            "1. Team A removed Mirage",
            "2. Team A removed Anubis",
            "3. Team B removed Nuke",
            "4. Team B removed Ancient",
            "5. Team B removed Inferno",
            "6. Team A removed Dust2",
            "7. Overpass was left over",
        ],
    }

    match = parse_veto_match(record, DEFAULT_ERAS)

    assert match is not None
    team_a_bans = [action for action in match.actions if action.team_id == "TEAM A" and action.action_type == "ban"]
    assert [action.team_ban_index for action in team_a_bans] == [1, 2, 3]
    assert [action.map_name for action in team_a_bans] == ["Mirage", "Anubis", "Dust2"]


def test_eventual_ban_signal_learns_bo1_later_permaban():
    history = VetoHistory(window_days=90)
    prior = parse_veto_match(
        {
            "url": "https://www.hltv.org/matches/1/a-vs-b",
            "date": "2026-05-01",
            "format": "bo1",
            "team1": "Team A",
            "team2": "Team B",
            "hltv_vetoes": [
                "1. Team A removed Mirage",
                "2. Team A removed Anubis",
                "3. Team B removed Nuke",
                "4. Team B removed Ancient",
                "5. Team B removed Inferno",
                "6. Team A removed Dust2",
                "7. Overpass was left over",
            ],
        },
        DEFAULT_ERAS,
    )
    target = parse_veto_match(
        {
            "url": "https://www.hltv.org/matches/2/a-vs-c",
            "date": "2026-05-02",
            "format": "bo1",
            "team1": "Team A",
            "team2": "Team C",
            "hltv_vetoes": [
                "1. Team A removed Ancient",
                "2. Team A removed Anubis",
                "3. Team C removed Nuke",
                "4. Team C removed Mirage",
                "5. Team C removed Inferno",
                "6. Team A removed Dust2",
                "7. Overpass was left over",
            ],
        },
        DEFAULT_ERAS,
    )

    history.update(prior)
    row = history.build_row(target.actions[1])
    probs = predict_probabilities(row, ModelSpec("eventual_only", {"eventual": 1.0}))

    assert probs["Anubis"] > probs["Overpass"]
    assert row.team_ban_index == 2


def test_normalize_name_uppercases_without_aliasing():
    assert normalize_name("  9z  ") == "9Z"


def test_randomized_veto_note_excludes_match():
    record = {
        "url": "https://www.hltv.org/matches/3/a-vs-b",
        "date": "2026-05-01",
        "format": "bo3",
        "team1": "EC BANGA",
        "team2": "Team B",
        "match_info": [
            "** EC BANGA's map bans and picks are randomized as they failed to show up in time for the VETO process."
        ],
        "hltv_vetoes": [
            "1. EC BANGA removed Mirage",
            "2. Team B removed Nuke",
            "3. EC BANGA picked Anubis",
            "4. Team B picked Ancient",
            "5. EC BANGA removed Dust2",
            "6. Team B removed Inferno",
            "7. Overpass was left over",
        ],
    }

    assert parse_veto_match(record, DEFAULT_ERAS) is None


def test_shared_lock_model_does_not_force_shared_first_ban():
    row = FeatureRow(
        actual_map="Mirage",
        pool=("Anubis", "Mirage", "Nuke"),
        match_format="bo3",
        team_ban_index=1,
        prior_team_bans=20,
        signals={
            "Anubis": {
                "slot": 0.8,
                "eventual": 0.8,
                "team_ban": 0.8,
                "raw_first_slot_rate": 0.8,
                "raw_first_slot_sample": 20.0,
                "opponent_raw_first_slot_rate": 0.85,
                "opponent_raw_first_slot_sample": 20.0,
            },
            "Mirage": {
                "slot": 0.1,
                "eventual": 0.1,
                "team_ban": 0.1,
                "raw_first_slot_rate": 0.1,
                "raw_first_slot_sample": 20.0,
                "opponent_raw_first_slot_rate": 0.0,
                "opponent_raw_first_slot_sample": 20.0,
            },
            "Nuke": {
                "slot": 0.1,
                "eventual": 0.1,
                "team_ban": 0.1,
                "raw_first_slot_rate": 0.1,
                "raw_first_slot_sample": 20.0,
                "opponent_raw_first_slot_rate": 0.0,
                "opponent_raw_first_slot_sample": 20.0,
            },
        },
    )
    locked = ModelSpec(
        "locked",
        {
            "slot": 0.6,
            "eventual": 0.25,
            "team_ban": 0.15,
            "_lock_probability": 0.9,
            "_lock_min_sample": 10,
            "_lock_min_rate": 0.75,
        },
    )
    shared = ModelSpec(
        "shared",
        {
            "slot": 0.6,
            "eventual": 0.25,
            "team_ban": 0.15,
            "_lock_probability": 0.9,
            "_lock_min_sample": 10,
            "_lock_min_rate": 0.75,
            "_shared_lock_min_rate": 0.75,
            "_shared_lock_min_sample": 10,
        },
    )

    assert predict_probabilities(row, locked)["Anubis"] == 0.9
    assert predict_probabilities(row, shared)["Anubis"] < 0.9
