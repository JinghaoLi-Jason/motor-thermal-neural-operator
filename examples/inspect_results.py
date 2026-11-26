"""Inspect and verify the published experiment artifacts without ML dependencies."""

import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def inspect(root):
    provenance = read_json(root / "artifacts/provenance.json")
    for relative_path, record in provenance["files"].items():
        path = root / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {relative_path}")

    model_root = root / "artifacts/pretrained/final_all"
    config = read_json(model_root / "config.json")
    normalization = read_json(model_root / "normalization.json")
    history = read_json(model_root / "history.json")
    metrics = read_json(root / "reports/varying_load/metrics.json")

    for model_key, metric_key, normalization_key in [
        ("dT_mean", "model_dT_mean", "fold_dT_mean"),
        ("dT_std", "model_dT_std", "fold_dT_std"),
    ]:
        if not math.isclose(config[model_key], metrics[metric_key], rel_tol=1e-9):
            raise ValueError(f"Model and metric normalization differ: {model_key}")
        if not math.isclose(config[model_key], normalization[normalization_key], rel_tol=1e-9):
            raise ValueError(f"Model and normalization record differ: {model_key}")

    with zipfile.ZipFile(model_root / "final_model.pth") as checkpoint:
        corrupt_member = checkpoint.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Corrupt checkpoint archive member: {corrupt_member}")

    print(f"Verified {len(provenance['files'])} original experiment artifacts.")
    print(f"Model parameters: {config['n_params']:,}")
    print(f"Training conditions: {', '.join(config['train_conditions'])}")
    print(f"Recorded training epochs: {len(history)}")
    print(f"Varying-load maximum-temperature RMSE: {metrics['T_max_rmse']:.4f} degC")
    print(f"Varying-load spatial-mean RMSE: {metrics['T_mean_rmse']:.4f} degC")
    print(f"Recorded full-profile inference: {metrics['inference_time_s']:.4f} s")
    print(f"Hardware: {metrics['device_info']['gpu_name']}")
    print("\nHeld-out condition | Field RMSE | Tmax RMSE | Tmean RMSE (degC)")

    with (root / "reports/cross_validation/cv_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        fold = read_json(root / "reports/cross_validation" / row["fold"] / "cv_inference_metrics.json")
        for column in ["field_rmse_avg_degC", "Tmax_rmse_degC", "Tmean_rmse_degC"]:
            if not math.isclose(float(row[column]), float(fold[column]), rel_tol=1e-9):
                raise ValueError(f"CV summary differs from {row['fold']}: {column}")
        print(
            f"{row['test_condition_label']:>18} | "
            f"{float(row['field_rmse_avg_degC']):10.3f} | "
            f"{float(row['Tmax_rmse_degC']):9.3f} | "
            f"{float(row['Tmean_rmse_degC']):10.3f}"
        )
    print("\nThese are saved experiment records, not a new inference run.")


if __name__ == "__main__":
    inspect(Path(__file__).resolve().parents[1])
