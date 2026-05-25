import json
import sqlite3

from evaluation import shadow_ledger


def test_model_version_evaluation_payload_uses_named_evaluations(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow_ledger, "DB_PATH", str(tmp_path / "shadow_ledger.db"))

    conn = shadow_ledger.get_db()
    try:
        conn.execute(
            """INSERT INTO model_versions
               (version_id, trained_at, best_val_loss, epochs_run, num_features,
                data_stats_json, test_brier_score, test_log_loss)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "v_test",
                "2026-05-20 16:15:12",
                0.58,
                12,
                17,
                '{"total_maps": 100, "total_matches": 50}',
                0.21,
                0.61,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    shadow_ledger.register_model_evaluation(
        version_id="v_test",
        eval_name="temporal_test",
        rows=20,
        matches=10,
        log_loss=0.62,
        label_mean=0.55,
        prediction_mean=0.52,
    )
    shadow_ledger.register_model_evaluation(
        version_id="v_test",
        eval_name="temporal_test",
        rows=21,
        matches=11,
        log_loss=0.63,
        label_mean=0.56,
        prediction_mean=0.53,
    )

    payload = shadow_ledger.model_version_payload()

    assert payload["versions"][0]["version_id"] == "v_test"
    assert len(payload["evaluations"]) == 1
    assert payload["evaluations"][0]["eval_name"] == "temporal_test"
    assert payload["evaluations"][0]["rows"] == 21
    assert payload["evaluations"][0]["log_loss"] == 0.63


def test_get_db_migrates_snapshot_forfeit_columns(tmp_path, monkeypatch):
    db_path = tmp_path / "shadow_ledger.db"
    monkeypatch.setattr(shadow_ledger, "DB_PATH", str(db_path))

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """CREATE TABLE snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_url TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )"""
        )
        conn.commit()
    finally:
        conn.close()

    conn = shadow_ledger.get_db()
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(snapshots)").fetchall()}
    finally:
        conn.close()

    assert "forfeit_prob" in columns
    assert "forfeit_model_metadata_json" in columns
    assert "polymarket_fair_prob_a" in columns
    assert "polymarket_fair_prob_b" in columns


def test_record_predictions_stores_polymarket_forfeit_adjusted_fair_probs(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow_ledger, "DB_PATH", str(tmp_path / "shadow_ledger.db"))

    class FakeForfeitContext:
        model_path = None
        metadata = {
            "trained_at": "2026-05-20 12:00:00",
            "model_type": "fake_model",
            "calibration_method": "none",
            "features": ["team1_rank"],
            "target": "forfeit_target",
        }

        def predict_forfeit_probability(self, match):
            assert match["team1"] == "Alpha"
            assert match["team2"] == "Beta"
            assert match["format"] == "bo3"
            return 0.2

    fake_ctx = FakeForfeitContext()
    monkeypatch.setattr(
        shadow_ledger,
        "_load_forfeit_predictor_context",
        lambda: (fake_ctx, shadow_ledger._forfeit_model_metadata(fake_ctx), None),
    )

    shadow_ledger.record_predictions(
        [
            {
                "match": {"url": "https://example.test/match/1", "date": "2026-05-26", "time": "18:00"},
                "team_a": "Alpha",
                "team_b": "Beta",
                "fmt": "bo3",
                "prob1": 0.55,
                "o1": 2.0,
                "o2": 3.0,
                "edge1": 5.0,
                "edge2": -5.0,
            }
        ],
        version_id="v_test",
    )

    conn = shadow_ledger.get_db()
    try:
        row = conn.execute(
            """SELECT forfeit_prob, forfeit_model_metadata_json,
                      polymarket_fair_prob_a, polymarket_fair_prob_b,
                      analytics_summary_json
               FROM snapshots"""
        ).fetchone()
    finally:
        conn.close()

    assert row["forfeit_prob"] == 0.2
    assert row["polymarket_fair_prob_a"] == 0.58
    assert row["polymarket_fair_prob_b"] == 0.42

    metadata = json.loads(row["forfeit_model_metadata_json"])
    assert metadata["status"] == "applied"
    assert metadata["adjustment_applied"] is True
    assert metadata["trained_at"] == "2026-05-20 12:00:00"

    analytics = json.loads(row["analytics_summary_json"])
    assert analytics["forfeit_adjustment"]["status"] == "applied"
