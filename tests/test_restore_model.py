import json

import pytest

from evaluation import shadow_ledger
from scripts import restore_model


def _insert_version(
    tmp_path,
    *,
    version_id: str,
    trained_at: str,
    weights_bytes: bytes,
    scaler_bytes: bytes,
    stats: dict,
    is_active: int = 0,
):
    archive_dir = tmp_path / "checkpoints" / "archive" / version_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    weights_path = archive_dir / "best_mvp_model.pt"
    scaler_path = archive_dir / "scaler.pkl"
    weights_path.write_bytes(weights_bytes)
    scaler_path.write_bytes(scaler_bytes)

    conn = shadow_ledger.get_db()
    try:
        conn.execute(
            """INSERT INTO model_versions
               (version_id, trained_at, best_val_loss, epochs_run, num_features,
                data_stats_json, weights_path, scaler_path,
                test_brier_score, test_log_loss, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version_id,
                trained_at,
                0.60,
                10,
                17,
                json.dumps(stats),
                str(weights_path),
                str(scaler_path),
                0.21,
                0.61,
                is_active,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_restore_version_backs_up_live_files_and_marks_active(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    monkeypatch.setattr(shadow_ledger, "DB_PATH", str(data_dir / "shadow_ledger.db"))
    monkeypatch.setattr(restore_model, "PROJECT_ROOT", tmp_path)

    _insert_version(
        tmp_path,
        version_id="v_old",
        trained_at="2026-09-02 15:32:36",
        weights_bytes=b"old-weights",
        scaler_bytes=b"old-scaler",
        stats={"training_date": "2026-09-02 15:25:18", "total_maps": 19096},
        is_active=0,
    )
    _insert_version(
        tmp_path,
        version_id="v_new",
        trained_at="2026-11-17 16:00:00",
        weights_bytes=b"new-weights",
        scaler_bytes=b"new-scaler",
        stats={"training_date": "2026-11-17 16:00:00", "total_maps": 18393},
        is_active=1,
    )

    (checkpoint_dir / "best_mvp_model.pt").write_bytes(b"new-weights")
    (checkpoint_dir / "scaler.pkl").write_bytes(b"new-scaler")
    (data_dir / "training_state.json").write_text('{"total_maps": 18393}', encoding="utf-8")

    assert shadow_ledger.get_latest_version_id() == "v_new"

    result = restore_model.restore_version(
        "v_old",
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
    )

    assert (checkpoint_dir / "best_mvp_model.pt").read_bytes() == b"old-weights"
    assert (checkpoint_dir / "scaler.pkl").read_bytes() == b"old-scaler"
    restored_state = json.loads((data_dir / "training_state.json").read_text(encoding="utf-8"))
    assert restored_state == {"training_date": "2026-09-02 15:25:18", "total_maps": 19096}
    assert shadow_ledger.get_latest_version_id() == "v_old"

    backup_dir = result["backup_dir"]
    assert backup_dir is not None
    assert (backup_dir / "best_mvp_model.pt").read_bytes() == b"new-weights"
    assert (backup_dir / "scaler.pkl").read_bytes() == b"new-scaler"
    assert (backup_dir / "training_state.json").read_text(encoding="utf-8") == '{"total_maps": 18393}'


def test_restore_version_fails_before_touching_live_files_when_archive_missing(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    monkeypatch.setattr(shadow_ledger, "DB_PATH", str(data_dir / "shadow_ledger.db"))
    monkeypatch.setattr(restore_model, "PROJECT_ROOT", tmp_path)

    (checkpoint_dir / "best_mvp_model.pt").write_bytes(b"keep-weights")

    with pytest.raises(ValueError, match="was not found"):
        restore_model.restore_version("v_missing", data_dir=data_dir, checkpoint_dir=checkpoint_dir)

    assert (checkpoint_dir / "best_mvp_model.pt").read_bytes() == b"keep-weights"
    assert not (checkpoint_dir / "backups").exists()
