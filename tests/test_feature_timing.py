import numpy as np
import pandas as pd
import pytest

from config import DEFAULT_SOS_RANK, DEFAULT_TEAM_RANK
from evaluation.feature_experiments import add_h2h_window_features
from model import predict
from processing.clean import repair_default_ranks
from processing.features import MODEL_FEATURES, compute_features


HISTORY_FEATURES = [*MODEL_FEATURES, "team_a_gen_matches_30d", "team_b_gen_matches_30d"]


def map_row(match, day, map_name, winner="A", *, opponent="B", picker=None, lan=False, forfeit=False):
    score_a, score_b = (13, 5) if winner == "A" else (8, 13)
    return {
        "match_id": match,
        "date": pd.Timestamp(f"2025-01-{day:02d}", tz="UTC"),
        "map_name": map_name,
        "team_a_id": "A",
        "team_b_id": opponent,
        "team_a_picked": picker == "A",
        "team_b_picked": picker == opponent,
        "team_a_world_rank": 5,
        "team_b_world_rank": 20 if opponent == "B" else 40,
        "match_has_forfeit": forfeit,
        "is_lan": lan,
        "winner_id": winner,
        "score_a": score_a,
        "score_b": score_b,
    }


def feature_rows(rows):
    return compute_features(pd.DataFrame(rows)).set_index(["match_id", "map_name"]).sort_index()


def test_map_results_only_affect_later_dates():
    rows = [
        map_row("prior", 1, "nuke", opponent="C", lan=True),
        map_row("bo3", 2, "mirage", picker="A"),
        map_row("bo3", 2, "ancient", picker="A"),
        map_row("bo3", 2, "nuke", "B", picker="B"),
        map_row("later", 3, "mirage", picker="B"),
        map_row("later", 3, "ancient"),
    ]
    before = feature_rows(rows)
    flipped = feature_rows([rows[0], {**rows[1], "winner_id": "B", "score_a": 11, "score_b": 13}, *rows[2:]])
    pd.testing.assert_frame_equal(
        before.loc[["prior", "bo3"], HISTORY_FEATURES],
        flipped.loc[["prior", "bo3"], HISTORY_FEATURES],
    )
    assert before.loc[("bo3", "mirage"), "label"] != flipped.loc[("bo3", "mirage"), "label"]
    assert before.loc[("bo3", "mirage"), "picker_diff"] == 1
    assert before.loc[("bo3", "nuke"), "picker_diff"] == -1
    assert before.loc[("bo3", "mirage"), "map_win_rate_diff"] == 0
    assert before.loc[("bo3", "nuke"), "map_win_rate_diff"] == pytest.approx(0.1)
    assert (before.loc["bo3", "team_a_gen_matches_30d"] == 1).all()
    assert (before.loc["bo3", "team_b_gen_matches_30d"] == 0).all()
    assert (before.loc["bo3", "lan_rate_diff"] == 1).all()
    assert before.loc[("bo3", "nuke"), "sos_diff"] == pytest.approx(np.log(DEFAULT_SOS_RANK / 40))

    later = before.loc[("later", "mirage")]
    changed_later = flipped.loc[("later", "mirage")]
    assert later["team_a_gen_matches_30d"] == 4
    assert later["team_b_gen_matches_30d"] == 3
    assert later["h2h_a_wins_30d"] == 2
    assert changed_later["h2h_a_wins_30d"] == 1
    assert later["h2h_b_wins_30d"] == 1
    assert later["team_a_win_streak"] == pytest.approx(np.log1p(2))
    assert changed_later["team_a_win_streak"] == 0
    assert later["lan_rate_diff"] == 0.25
    assert later["map_comfort_diff"] == 1
    assert before.loc[("later", "ancient"), "map_comfort_diff"] == 0
    for feature in (
        "win_rate_7d_diff", "win_rate_30d_diff", "win_rate_90d_diff",
        "map_win_rate_diff", "dominance_diff", "resilience_diff",
    ):
        assert later[feature] != changed_later[feature]


def test_same_day_series_share_history_and_streaks_are_order_independent(tmp_path, monkeypatch):
    rows = [
        map_row("prior", 1, "mirage"),
        map_row("win", 2, "nuke"),
        map_row("loss", 2, "ancient", "B"),
        map_row("first-win", 3, "mirage"),
        map_row("second-win", 3, "ancient"),
        map_row("forfeit", 3, "nuke", "B", forfeit=True),
        map_row("later", 4, "mirage"),
    ]
    before = feature_rows(rows)
    pd.testing.assert_frame_equal(before, feature_rows(list(reversed(rows))))
    assert (before.loc[["win", "loss"], "team_a_win_streak"] == np.log1p(1)).all()
    assert before.loc[("win", "nuke"), "h2h_b_wins_30d"] == 0
    assert before.loc[("loss", "ancient"), "h2h_a_wins_30d"] == 1
    assert before.loc[("first-win", "mirage"), "team_a_win_streak"] == 0
    assert before.loc[("later", "mirage"), "team_a_win_streak"] == np.log1p(2)
    assert before.loc[("later", "mirage"), "h2h_b_wins_30d"] == 2

    pd.DataFrame(rows[:-1]).to_parquet(tmp_path / "clean_maps.parquet")
    monkeypatch.setattr(predict, "PROCESSED_DIR", tmp_path)
    assert predict.load_latest_state()[4] == {"A": 2, "B": 0}


def test_experimental_h2h_uses_strictly_earlier_dates():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-05", "2025-01-02", "2025-01-02", "2025-01-02", "2025-01-01", "2024-12-01"], utc=True),
        "match_id": ["later", "bo3", "bo3", "same-day", "prior", "expired"],
        "team_a_id": ["B", "A", "A", "B", "A", "A"],
        "team_b_id": ["A", "B", "B", "A", "B", "B"],
        "label": [1, 1, 0, 0, 1, 1],
    }, index=[19, 5, 4, 2, 8, 9])
    columns = ["h2h_a_wins_14d", "h2h_b_wins_14d"]
    enriched = add_h2h_window_features(frame, 14)
    assert enriched[columns].values.tolist() == [[1, 3], [1, 0], [1, 0], [0, 1], [0, 0], [0, 0]]
    changed = frame.copy()
    changed.loc[5, "label"] = 0
    rebuilt = add_h2h_window_features(changed.iloc[::-1], 14)
    pd.testing.assert_frame_equal(enriched.loc[[5, 4, 2, 8, 9], columns], rebuilt.loc[[5, 4, 2, 8, 9], columns])
    assert rebuilt.loc[19, columns].tolist() == [2, 2]


def test_missing_rank_repair_uses_earlier_dates():
    rows = [
        map_row("prior", 1, "mirage"),
        {**map_row("known", 2, "mirage"), "team_a_world_rank": 9},
        map_row("missing", 2, "nuke"),
        map_row("later", 3, "nuke"),
    ]
    frame = pd.DataFrame(rows).set_index("match_id")
    frame["team_a_world_rank_missing"] = frame.index.isin(["missing", "later"])
    frame.loc[frame["team_a_world_rank_missing"], "team_a_world_rank"] = DEFAULT_TEAM_RANK
    repaired = repair_default_ranks(frame)
    assert repaired.loc["missing", "team_a_world_rank"] == 5
    assert repaired.loc["later", "team_a_world_rank"] == 9


@pytest.mark.parametrize("source", ["world", "vrs"])
def test_rank_repair_follows_team_across_sides_and_ignores_same_day_conflicts(source):
    def rank_row(match, day, a, b, rank_a, rank_b):
        return {
            "match_id": match,
            "date": pd.Timestamp(f"2025-01-{day:02d}", tz="UTC"),
            "team_a_id": a,
            "team_b_id": b,
            f"team_a_{source}_rank": rank_a,
            f"team_b_{source}_rank": rank_b,
            f"team_a_{source}_rank_missing": rank_a == DEFAULT_TEAM_RANK,
            f"team_b_{source}_rank_missing": rank_b == DEFAULT_TEAM_RANK,
        }

    frame = pd.DataFrame([
        rank_row("prior", 1, "A", "B", 5, 20),
        rank_row("swapped", 2, "B", "A", DEFAULT_TEAM_RANK, DEFAULT_TEAM_RANK),
        rank_row("observed-a", 2, "A", "C", 9, 30),
        rank_row("observed-b", 2, "C", "A", 30, 11),
        rank_row("later", 3, "B", "A", DEFAULT_TEAM_RANK, DEFAULT_TEAM_RANK),
    ]).set_index("match_id")
    repaired = repair_default_ranks(frame)
    for match in ["swapped", "later"]:
        assert repaired.loc[match, f"team_a_{source}_rank"] == 20
        assert repaired.loc[match, f"team_b_{source}_rank"] == 5
    pd.testing.assert_frame_equal(
        repaired.sort_index(), repair_default_ranks(frame.iloc[::-1]).sort_index()
    )


def test_later_results_and_context_cannot_change_earlier_features():
    rows = [
        map_row("prior", 1, "mirage", picker="A"),
        map_row("target", 2, "mirage", picker="B"),
        map_row("target", 2, "nuke", "B"),
        map_row("future", 3, "mirage", picker="A"),
    ]
    changed = [*rows[:-1], {
        **rows[-1], "winner_id": "B", "score_a": 0, "score_b": 13,
        "team_a_world_rank": 100, "team_b_world_rank": 1,
        "is_lan": True, "team_a_picked": False, "team_b_picked": True,
    }]
    pd.testing.assert_frame_equal(
        feature_rows(rows).loc[["prior", "target"], HISTORY_FEATURES],
        feature_rows(changed).loc[["prior", "target"], HISTORY_FEATURES],
    )
