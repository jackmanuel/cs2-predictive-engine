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
