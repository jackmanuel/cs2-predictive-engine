import json
from pathlib import Path

from model import veto_sim


def test_load_team_ban_history_excludes_randomized_vetoes(tmp_path):
    raw_path = tmp_path / "hltv_matches.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "url": "https://www.hltv.org/matches/1/a-vs-b",
                    "date": "2026-05-01",
                    "format": "bo3",
                    "team1": "Team A",
                    "team2": "Team B",
                    "match_info": [
                        "** Team A's map bans and picks are randomized as they failed to show up in time for the VETO process."
                    ],
                    "hltv_vetoes": [
                        "1. Team A removed Anubis",
                        "2. Team B removed Nuke",
                    ],
                },
                {
                    "url": "https://www.hltv.org/matches/2/a-vs-c",
                    "date": "2026-05-02",
                    "format": "bo3",
                    "team1": "Team A",
                    "team2": "Team C",
                    "hltv_vetoes": [
                        "1. Team A removed Mirage",
                        "2. Team C removed Nuke",
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    history = veto_sim.load_team_ban_history("TEAM A", raw_path=raw_path)

    assert history["series"] == 1
    assert history["slot_counts"][("bo3", 1)]["Mirage"] == 1
    assert history["slot_counts"][("bo3", 1)]["Anubis"] == 0


def test_ban_weights_use_slot_and_eventual_ban_history():
    stats = {"metadata": {"ban_slot_totals": {"bo1:1": 1, "bo1:2": 1}, "eventual_ban_totals": {"bo1": 2}, "team_ban_total": 2}}
    for map_name in veto_sim.MAP_POOL:
        stats[map_name] = {
            "ban_slot_counts": {},
            "eventual_ban_counts": {},
            "team_ban_count": 0,
        }
    stats["Mirage"]["ban_slot_counts"]["bo1:1"] = 1
    stats["Mirage"]["eventual_ban_counts"]["bo1"] = 1
    stats["Mirage"]["team_ban_count"] = 1
    stats["Anubis"]["ban_slot_counts"]["bo1:2"] = 1
    stats["Anubis"]["eventual_ban_counts"]["bo1"] = 1
    stats["Anubis"]["team_ban_count"] = 1

    first_slot_weights = dict(
        zip(
            veto_sim.MAP_POOL,
            veto_sim.get_ban_weight(stats, stats, veto_sim.MAP_POOL, series_format="bo1", team_ban_index=1),
        )
    )
    second_slot_weights = dict(
        zip(
            veto_sim.MAP_POOL,
            veto_sim.get_ban_weight(stats, stats, veto_sim.MAP_POOL, series_format="bo1", team_ban_index=2),
        )
    )

    assert first_slot_weights["Mirage"] > first_slot_weights["Anubis"]
    assert second_slot_weights["Anubis"] > second_slot_weights["Mirage"]
    assert second_slot_weights["Anubis"] > second_slot_weights["Cache"]


def test_historical_overpass_ban_preserves_slots_but_is_not_an_active_candidate(tmp_path):
    raw_path = tmp_path / "hltv_matches.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "url": "https://www.hltv.org/matches/3/a-vs-b",
                    "date": "2026-07-01",
                    "format": "bo3",
                    "team1": "Team A",
                    "team2": "Team B",
                    "hltv_vetoes": [
                        "1. Team A removed Overpass",
                        "2. Team B removed Nuke",
                        "3. Team A removed Mirage",
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    history = veto_sim.load_team_ban_history("TEAM A", raw_path=raw_path)

    assert "Overpass" not in veto_sim.MAP_POOL
    assert history["slot_counts"][("bo3", 2)]["Mirage"] == 1
    assert history["team_ban_counts"]["Overpass"] == 0


def test_high_sample_first_ban_history_locks_permaban():
    stats = {
        "metadata": {
            "ban_slot_totals": {"bo3:1": 20},
            "eventual_ban_totals": {"bo3": 40},
            "team_ban_total": 40,
        }
    }
    for map_name in veto_sim.MAP_POOL:
        stats[map_name] = {
            "ban_slot_counts": {"bo3:1": 0},
            "eventual_ban_counts": {"bo3": 0},
            "team_ban_count": 0,
        }
    stats["Anubis"]["ban_slot_counts"]["bo3:1"] = 16
    stats["Anubis"]["eventual_ban_counts"]["bo3"] = 16
    stats["Anubis"]["team_ban_count"] = 16
    stats["Mirage"]["ban_slot_counts"]["bo3:1"] = 4
    stats["Mirage"]["eventual_ban_counts"]["bo3"] = 20
    stats["Mirage"]["team_ban_count"] = 20

    opponent_stats = {
        "metadata": {
            "ban_slot_totals": {"bo3:1": 20},
            "eventual_ban_totals": {"bo3": 20},
            "team_ban_total": 20,
        }
    }
    for map_name in veto_sim.MAP_POOL:
        opponent_stats[map_name] = {
            "ban_slot_counts": {"bo3:1": 0},
            "eventual_ban_counts": {"bo3": 0},
            "team_ban_count": 0,
        }
    opponent_stats["Nuke"]["ban_slot_counts"]["bo3:1"] = 16
    opponent_stats["Nuke"]["eventual_ban_counts"]["bo3"] = 16
    opponent_stats["Nuke"]["team_ban_count"] = 16
    opponent_stats["Mirage"]["ban_slot_counts"]["bo3:1"] = 4
    opponent_stats["Mirage"]["eventual_ban_counts"]["bo3"] = 4
    opponent_stats["Mirage"]["team_ban_count"] = 4

    weights = dict(
        zip(
            veto_sim.MAP_POOL,
            veto_sim.get_ban_weight(stats, opponent_stats, veto_sim.MAP_POOL, series_format="bo3", team_ban_index=1),
        )
    )

    assert weights["Anubis"] == veto_sim.LOCKED_FIRST_BAN_PROBABILITY
    assert sum(weights.values()) == 1.0


def test_shared_permaban_suppresses_first_ban_lock():
    team_stats = {
        "metadata": {
            "ban_slot_totals": {"bo3:1": 20},
            "eventual_ban_totals": {"bo3": 20},
            "team_ban_total": 20,
        }
    }
    opponent_stats = {
        "metadata": {
            "ban_slot_totals": {"bo3:1": 20},
            "eventual_ban_totals": {"bo3": 20},
            "team_ban_total": 20,
        }
    }
    for stats in (team_stats, opponent_stats):
        for map_name in veto_sim.MAP_POOL:
            stats[map_name] = {
                "ban_slot_counts": {"bo3:1": 0},
                "eventual_ban_counts": {"bo3": 0},
                "team_ban_count": 0,
            }
        stats["Anubis"]["ban_slot_counts"]["bo3:1"] = 16
        stats["Anubis"]["eventual_ban_counts"]["bo3"] = 16
        stats["Anubis"]["team_ban_count"] = 16
        stats["Mirage"]["ban_slot_counts"]["bo3:1"] = 4
        stats["Mirage"]["eventual_ban_counts"]["bo3"] = 4
        stats["Mirage"]["team_ban_count"] = 4

    weights = dict(
        zip(
            veto_sim.MAP_POOL,
            veto_sim.get_ban_weight(team_stats, opponent_stats, veto_sim.MAP_POOL, series_format="bo3", team_ban_index=1),
        )
    )

    assert weights["Anubis"] != veto_sim.LOCKED_FIRST_BAN_PROBABILITY
    assert weights["Anubis"] > weights["Mirage"]
