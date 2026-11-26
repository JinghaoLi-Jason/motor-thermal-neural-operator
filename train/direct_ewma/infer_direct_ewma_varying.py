"""Evaluate the final direct EWMA FNO on the 100%-50%-75% varying-load case."""

import argparse
import json
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from neuralop.models import FNO
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--dataset", default="data/processed/fno_6cond_element_512_direct_ewma.h5")
    p.add_argument("--val_csv", default="data/raw/NodeTemperature_100_50_75_correct_sampling_time.csv")
    p.add_argument("--coord_file", default="data/raw/coordinate_exp_test.txt")
    p.add_argument("--output_dir", default="results/direct_ewma_varying_eval")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--t_end", type=float, default=60000.0)
    p.add_argument("--gt_cache", default="cache/varying_load_gt_timeseries_512.npz")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 8,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def get_device_info(device):
    info = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "pytorch_version": str(torch.__version__),
        "pytorch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update({
            "gpu_index": int(index),
            "gpu_name": torch.cuda.get_device_name(index),
            "gpu_total_memory_gib": float(props.total_memory) / 1024**3,
        })
    else:
        info.update({"gpu_index": None, "gpu_name": None, "gpu_total_memory_gib": None})
    return info


def get_load_at_time(t):
    if t < 19000.0:
        return 1.0
    if t < 39000.0:
        return 0.5
    return 0.75


def alpha_from_span(span):
    return 2.0 / (span + 1.0)


def load_model_and_meta(model_dir, dataset, device):
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing model configuration: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    if "dT_mean" not in config or "dT_std" not in config:
        raise KeyError("config.json must contain the dT_mean and dT_std used during training.")

    with h5py.File(dataset, "r") as f:
        mask = f["mask"][:].astype(np.float32)
        x_grid = f["x_grid"][:]
        y_grid = f["y_grid"][:]
        X_sample = f["X_all"][0]
        in_channels = int(f["X_all"].shape[1])
        meta = {
            "mask": mask,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "in_channels": in_channels,
            "T_abs_mean": float(f.attrs["T_abs_mean"]),
            "T_abs_std": float(f.attrs["T_abs_std"]),
            "dT_mean": float(config["dT_mean"]),
            "dT_std": float(config["dT_std"]),
            "dataset_dT_mean": float(f.attrs["dT_mean"]),
            "dataset_dT_std": float(f.attrs["dT_std"]),
            "normalization_source": config.get("normalization", {}).get("source", "unknown"),
            "ewma_spans": json.loads(f.attrs["ewma_spans_json"]),
            "ewma_alignment": str(
                f.attrs.get("ewma_alignment", "missing")
            ),
            "heat_source_mode": str(
                f.attrs.get("heat_source_mode", "missing")
            ),
            "Q_input_definition": str(
                f.attrs.get("Q_input_definition", "missing")
            ),

        }
        if meta["ewma_alignment"] != "previous":
            raise ValueError(
                "This inference script expects previous-aligned EWMA "
                f"features, but the dataset contains "
                f"ewma_alignment={meta['ewma_alignment']!r}."
            )

    expected_channels = int(config.get("input_channels", in_channels))
    if in_channels != expected_channels:
        raise ValueError(f"Input-channel mismatch: dataset={in_channels}, model={expected_channels}")

    model = FNO(
        n_modes=(config["modes"], config["modes"]),
        in_channels=in_channels,
        out_channels=1,
        hidden_channels=config["width"],
        n_layers=config["n_layers"],
    ).to(device)
    model_path = os.path.join(model_dir, "final_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing final model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    meta["x_coord"] = torch.tensor(X_sample[2], dtype=torch.float32, device=device)
    meta["y_coord"] = torch.tensor(X_sample[3], dtype=torch.float32, device=device)
    meta["Q_base_norm"] = torch.tensor(X_sample[4], dtype=torch.float32, device=device)
    return model, config, meta


def load_coordinates(path):
    try:
        data = np.loadtxt(path, skiprows=1)
    except Exception:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0].astype(int), data[:, 1:3].astype(float)


def rasterize_csv_at_time(df, t_target, coord_file, x_grid, y_grid, mask):
    time_col, node_col, temp_col = df.columns[:3]
    unique_times = np.sort(df[time_col].unique())
    t_actual = unique_times[np.argmin(np.abs(unique_times - t_target))]
    frame = df[df[time_col] == t_actual]
    coord_ids, coord_xy = load_coordinates(coord_file)
    id_to_xy = {nid: coord_xy[i] for i, nid in enumerate(coord_ids)}
    positions, temperatures = [], []
    for nid, temp in zip(frame[node_col].astype(int), frame[temp_col].astype(float)):
        if nid in id_to_xy:
            positions.append(id_to_xy[nid])
            temperatures.append(temp)
    positions = np.asarray(positions)
    temperatures = np.asarray(temperatures)
    xx, yy = np.meshgrid(x_grid, y_grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    field = griddata(positions, temperatures, points, method="linear").reshape(xx.shape)
    if np.any(np.isnan(field)):
        nearest = griddata(positions, temperatures, points, method="nearest").reshape(xx.shape)
        field = np.where(np.isnan(field), nearest, field)
    return field * mask, float(t_actual)


def get_ground_truth_timeseries(df, coord_file, x_grid, y_grid, mask):
    times = np.sort(df[df.columns[0]].unique()).astype(float)
    active = mask > 0
    tmax, tmean = [], []
    print(f"Rasterizing GT at {len(times)} times...")
    for i, t in enumerate(times):
        field, _ = rasterize_csv_at_time(df, t, coord_file, x_grid, y_grid, mask)
        tmax.append(float(field[active].max()))
        tmean.append(float(field[active].mean()))
        if (i + 1) % 50 == 0 or i == len(times) - 1:
            print(f"  {i + 1}/{len(times)} t={t:.0f}s Tmax={tmax[-1]:.1f}")
    return times, np.asarray(tmax), np.asarray(tmean)


def load_or_create_gt_cache(df, coord_file, x_grid, y_grid, mask, cache_path):
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_path):
        print(f"Loading cached ground truth: {cache_path}")
        data = np.load(cache_path)
        return data["times"], data["T_max"], data["T_mean"]
    values = get_ground_truth_timeseries(df, coord_file, x_grid, y_grid, mask)
    np.savez_compressed(cache_path, times=values[0], T_max=values[1], T_mean=values[2])
    print(f"Saved ground-truth cache: {cache_path}")
    return values


def make_input(meta, initial_field, t, t_end, ewma_vals, device):
    mask = meta["mask"]
    h, w = mask.shape
    initial_norm = (initial_field - meta["T_abs_mean"]) / meta["T_abs_std"]
    initial_tensor = torch.tensor(initial_norm * mask, dtype=torch.float32, device=device)
    p = get_load_at_time(float(t))
    channels = [
        initial_tensor,
        torch.full((h, w), np.clip(float(t) / float(t_end), 0.0, 1.0), device=device),
        meta["x_coord"], meta["y_coord"], meta["Q_base_norm"],
        torch.full((h, w), p, device=device),
    ]
    channels.extend(torch.full((h, w), float(v), device=device) for v in ewma_vals)
    return torch.stack(channels).unsqueeze(0)


@torch.no_grad()
def predict_field(model, meta, initial_field, t, t_end, ewma_vals, device):
    pred_norm = model(make_input(meta, initial_field, t, t_end, ewma_vals, device))
    dT = pred_norm[0, 0].detach().cpu().numpy()
    dT = (dT * meta["dT_std"] + meta["dT_mean"]) * meta["mask"]
    return initial_field + dT


@torch.no_grad()
def predict_direct_times(
    model,
    meta,
    initial_field,
    times,
    t_end,
    device,
):
    active = meta["mask"] > 0

    # At the first sample, no preceding load has been observed.
    ewma_vals = np.zeros(
        len(meta["ewma_spans"]),
        dtype=np.float32,
    )

    memory_times = []
    loads = []

    memory = {
        f"ewma_{span}": []
        for span in meta["ewma_spans"]
    }

    pred_tmax = []
    pred_tmean = []
    fields = {}

    for i, t_value in enumerate(times):
        t = float(t_value)
        p = get_load_at_time(t)

        # -----------------------------------------------------
        # Previous alignment:
        # ewma_vals summarizes loads before the current sample.
        # Store the exact values supplied to the model.
        # -----------------------------------------------------
        memory_times.append(t)
        loads.append(p)

        for j, span in enumerate(meta["ewma_spans"]):
            memory[f"ewma_{span}"].append(
                float(ewma_vals[j])
            )

        # Current load p is supplied separately inside make_input().
        # EWMA contains only the preceding load history.
        field = predict_field(
            model,
            meta,
            initial_field,
            t,
            t_end,
            ewma_vals,
            device,
        )

        pred_tmax.append(
            float(field[active].max())
        )
        pred_tmean.append(
            float(field[active].mean())
        )

        if i % 30 == 0 or i == len(times) - 1:
            fields[t] = field.copy()

        # -----------------------------------------------------
        # Update memory after prediction so that the current
        # load contributes to the next sample.
        # -----------------------------------------------------
        for j, span in enumerate(meta["ewma_spans"]):
            alpha = alpha_from_span(span)
            ewma_vals[j] = (
                alpha * p
                + (1.0 - alpha) * ewma_vals[j]
            )

    load_memory = {
        "time_s": np.asarray(memory_times),
        "p": np.asarray(loads),
        "ewma": {
            key: np.asarray(values)
            for key, values in memory.items()
        },
    }

    return (
        np.asarray(pred_tmax),
        np.asarray(pred_tmean),
        fields,
        load_memory,
    )


def compute_ewma_until_time(times, target, spans):
    """
    Compute the EWMA state supplied to the model at target time.

    With previous alignment, only loads at times strictly before
    the target time contribute to the memory.
    """
    values = np.zeros(
        len(spans),
        dtype=np.float32,
    )

    for t_value in times:
        t = float(t_value)

        # Do not include the load at the target time.
        if t >= float(target):
            break

        p = get_load_at_time(t)

        for j, span in enumerate(spans):
            alpha = alpha_from_span(span)
            values[j] = (
                alpha * p
                + (1.0 - alpha) * values[j]
            )

    return values


def save_load_memory(load_memory, output_dir, dpi):
    data = {"time_s": load_memory["time_s"], "time_ks": load_memory["time_s"] / 1000, "p": load_memory["p"]}
    data.update(load_memory["ewma"])
    pd.DataFrame(data).to_csv(os.path.join(output_dir, "load_ewma_memory_features.csv"), index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot(data["time_ks"], data["p"], "k-", linewidth=2, label="Current load $p(t)$")
    for key, values in load_memory["ewma"].items():
        ax.plot(data["time_ks"], values, label=key.replace("_", " "))
    for switch in (19, 39):
        ax.axvline(switch, color="gray", linestyle=":")
    ax.set(xlabel="Time (ks)", ylabel="Load / memory value", title="Load input and EWMA memory features")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "load_ewma_memory_features.png"), dpi=dpi)
    fig.savefig(os.path.join(output_dir, "load_ewma_memory_features.pdf"))
    plt.close(fig)


def plot_prediction_timeseries(times, true_max, true_mean, pred_max, pred_mean, output_dir, rmse_max, rmse_mean, dpi):
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.2), sharex=True)
    t = times / 1000
    axes[0].plot(t, true_max, "k--", label="FEM $T_{max}$"); axes[0].plot(t, pred_max, "r-", label="Pred. $T_{max}$")
    axes[1].plot(t, true_mean, "k--", label="FEM $T_{mean}$"); axes[1].plot(t, pred_mean, "b-", label="Pred. $T_{mean}$")
    axes[2].plot(t, pred_max - true_max, "r-", label="$T_{max}$ error"); axes[2].plot(t, pred_mean - true_mean, "b-", label="$T_{mean}$ error")
    axes[2].axhline(0, color="black", linewidth=.7)
    axes[0].set_ylabel("$T_{max}$ (°C)"); axes[1].set_ylabel("$T_{mean}$ (°C)"); axes[2].set_ylabel("Error (°C)"); axes[2].set_xlabel("Time (ks)")
    for ax in axes:
        for switch in (19, 39): ax.axvline(switch, color="gray", linestyle=":")
        ax.grid(alpha=.3); ax.legend()
    fig.suptitle(f"Direct EWMA FNO varying-load prediction\n$T_{{max}}$ RMSE={rmse_max:.2f}°C, $T_{{mean}}$ RMSE={rmse_mean:.2f}°C")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "direct_ewma_timeseries.png"), dpi=dpi)
    fig.savefig(os.path.join(output_dir, "direct_ewma_timeseries.pdf"))
    plt.close(fig)


def plot_field_comparison(model, meta, df, coord_file, initial_field, times, output_dir, device, t_end, dpi):
    key_times = [15000, 25000, 45000, 60000]
    active = meta["mask"] > 0
    extent = [meta["x_grid"][0], meta["x_grid"][-1], meta["y_grid"][0], meta["y_grid"][-1]]
    preds, truths = [], []
    for t in key_times:
        ewma = compute_ewma_until_time(times, t, meta["ewma_spans"])
        preds.append(predict_field(model, meta, initial_field, t, t_end, ewma, device))
        truths.append(rasterize_csv_at_time(df, t, coord_file, meta["x_grid"], meta["y_grid"], meta["mask"])[0])
    preds, truths = np.asarray(preds), np.asarray(truths)
    errors = preds - truths
    tmin = min(preds[:, active].min(), truths[:, active].min()); tmax = max(preds[:, active].max(), truths[:, active].max())
    elim = max(1.0, float(np.max(np.abs(errors[:, active]))))
    fig, axes = plt.subplots(3, len(key_times), figsize=(7.16, 4.2))
    for col, t in enumerate(key_times):
        im0 = axes[0, col].imshow(np.where(active, preds[col], np.nan), origin="lower", extent=extent, cmap="jet", vmin=tmin, vmax=tmax, aspect="equal")
        axes[1, col].imshow(np.where(active, truths[col], np.nan), origin="lower", extent=extent, cmap="jet", vmin=tmin, vmax=tmax, aspect="equal")
        im2 = axes[2, col].imshow(np.where(active, errors[col], np.nan), origin="lower", extent=extent, cmap="RdBu_r", vmin=-elim, vmax=elim, aspect="equal")
        axes[0, col].set_title(f"t={t/1000:.0f} ks")
        axes[2, col].set_title(f"MAE={np.mean(np.abs(errors[col][active])):.2f}°C")
        for row in range(3): axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("Prediction"); axes[1, 0].set_ylabel("FEM"); axes[2, 0].set_ylabel("Error")
    fig.subplots_adjust(right=.90, hspace=.25, wspace=.04)
    fig.colorbar(im0, cax=fig.add_axes([.915, .39, .012, .48]), label="Temperature (°C)")
    fig.colorbar(im2, cax=fig.add_axes([.915, .08, .012, .22]), label="Error (°C)")
    fig.savefig(os.path.join(output_dir, "direct_ewma_field_comparison.png"), dpi=dpi)
    fig.savefig(os.path.join(output_dir, "direct_ewma_field_comparison.pdf"))
    plt.close(fig)


def select_monitoring_points(df, coord_file, x_grid, y_grid, mask, n_points=8, drop_first=2, select_time=49200.0):
    time_col, node_col, temp_col = df.columns[:3]
    node_ids, node_xy = load_coordinates(coord_file)
    tree = cKDTree(node_xy)
    iy, ix = np.where(mask > 0)
    keep = (iy % 8 == 0) & (ix % 8 == 0)
    iy, ix = iy[keep], ix[keep]
    nearest = tree.query(np.column_stack([x_grid[ix], y_grid[iy]]))[1]
    nearest_ids = node_ids[nearest]
    times = np.sort(df[time_col].unique()); tref = times[np.argmin(np.abs(times - select_time))]
    lookup = dict(zip(df[df[time_col] == tref][node_col].astype(int), df[df[time_col] == tref][temp_col].astype(float)))
    temps = np.asarray([lookup.get(int(n), np.nan) for n in nearest_ids])
    valid = np.isfinite(temps); iy, ix, nearest_ids, temps = iy[valid], ix[valid], nearest_ids[valid], temps[valid]
    selected, used = [], set()
    for q in np.linspace(.05, .95, n_points):
        for choice in np.argsort(np.abs(temps - np.quantile(temps, q))):
            if int(nearest_ids[choice]) not in used:
                used.add(int(nearest_ids[choice])); break
        selected.append({"label": f"P{len(selected)+1}", "ix": int(ix[choice]), "iy": int(iy[choice]), "x": float(x_grid[ix[choice]]), "y": float(y_grid[iy[choice]]), "node_id": int(nearest_ids[choice]), "T_ref": float(temps[choice]), "t_ref": float(tref)})
    selected = selected[drop_first:]
    for i, point in enumerate(selected, 1): point["label"] = f"P{i}"
    return selected

def plot_selected_monitoring_points(
    df,
    coord_file,
    meta,
    points,
    output_dir,
    dpi,
):
    """
    Plot the spatial locations used for point-wise temperature and
    error traces.

    The background shows the FEM temperature field at the reference
    time used to select the monitoring points.
    """
    if not points:
        raise ValueError("No monitoring points were selected.")

    # All selected points use the same reference time.
    reference_time = float(points[0]["t_ref"])

    reference_field, actual_time = rasterize_csv_at_time(
        df,
        reference_time,
        coord_file,
        meta["x_grid"],
        meta["y_grid"],
        meta["mask"],
    )

    active = meta["mask"] > 0

    extent = [
        meta["x_grid"][0],
        meta["x_grid"][-1],
        meta["y_grid"][0],
        meta["y_grid"][-1],
    ]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    image = ax.imshow(
        np.where(active, reference_field, np.nan),
        origin="lower",
        extent=extent,
        cmap="jet",
        aspect="equal",
    )

    # Plot the selected monitoring points.
    point_x = [point["x"] for point in points]
    point_y = [point["y"] for point in points]

    ax.scatter(
        point_x,
        point_y,
        s=42,
        facecolors="white",
        edgecolors="black",
        linewidths=0.9,
        zorder=3,
    )

    # Add P1, P2, ... labels.
    for point in points:
        ax.annotate(
            point["label"],
            (point["x"], point["y"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
                "linewidth": 0.5,
            },
            zorder=4,
        )

    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_title(
        "Selected monitoring-point locations\n"
        f"FEM temperature field at t = {actual_time / 1000:.1f} ks"
    )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Temperature (°C)")

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "selected_monitoring_points.png",
        ),
        dpi=dpi,
    )

    fig.savefig(
        os.path.join(
            output_dir,
            "selected_monitoring_points.pdf",
        )
    )

    plt.close(fig)

    # Save the point definitions to a separate CSV.
    point_table = pd.DataFrame(
        [
            {
                "label": point["label"],
                "node_id": point["node_id"],
                "x": point["x"],
                "y": point["y"],
                "grid_ix": point["ix"],
                "grid_iy": point["iy"],
                "selection_temperature_degC": point["T_ref"],
                "selection_time_s": point["t_ref"],
            }
            for point in points
        ]
    )

    point_table.to_csv(
        os.path.join(
            output_dir,
            "selected_monitoring_points.csv",
        ),
        index=False,
    )

def get_true_point_traces(df, points):
    time_col, node_col, temp_col = df.columns[:3]
    ids = [p["node_id"] for p in points]
    sub = df[df[node_col].astype(int).isin(ids)][[time_col, node_col, temp_col]].copy()
    sub[node_col] = sub[node_col].astype(int)
    pivot = sub.pivot_table(index=time_col, columns=node_col, values=temp_col, aggfunc="mean").sort_index()
    return pivot.index.to_numpy(float), {p["label"]: pivot[p["node_id"]].to_numpy(float) for p in points}


@torch.no_grad()
def predict_point_traces(
    model,
    meta,
    initial_field,
    times,
    points,
    t_end,
    device,
):
    traces = {
        point["label"]: []
        for point in points
    }

    ewma = np.zeros(
        len(meta["ewma_spans"]),
        dtype=np.float32,
    )

    for t_value in times:
        t = float(t_value)
        p_now = get_load_at_time(t)

        # Predict using preceding memory.
        field = predict_field(
            model,
            meta,
            initial_field,
            t,
            t_end,
            ewma,
            device,
        )

        for point in points:
            traces[point["label"]].append(
                float(
                    field[
                        point["iy"],
                        point["ix"],
                    ]
                )
            )

        # Update with the current load for the next time point.
        for j, span in enumerate(meta["ewma_spans"]):
            alpha = alpha_from_span(span)
            ewma[j] = (
                alpha * p_now
                + (1.0 - alpha) * ewma[j]
            )

    return {
        key: np.asarray(values)
        for key, values in traces.items()
    }

def validate_first_ewma_state(times, load_memory):
    if len(times) == 0:
        raise ValueError("Cannot validate an empty time sequence.")

    for key, values in load_memory["ewma"].items():
        if not np.isclose(values[0], 0.0):
            raise ValueError(
                f"Previous-alignment error: first {key} "
                f"value is {values[0]}, expected 0."
            )

def plot_point_outputs(df, coord_file, meta, points, times, true_traces, pred_traces, output_dir, dpi):
    metrics = []
    ncols, nrows = 2, int(np.ceil(len(points) / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, 1.75 * nrows), sharex=True); axes = np.asarray(axes).ravel()
    fig_err, ax_err = plt.subplots(figsize=(7.16, 3.2))
    for i, point in enumerate(points):
        label = point["label"]; true = true_traces[label]; pred = pred_traces[label]; err = pred - true
        row = {**point, "rmse_degC": float(np.sqrt(np.mean(err**2))), "mae_degC": float(np.mean(np.abs(err))), "max_err_degC": float(np.max(np.abs(err)))}; metrics.append(row)
        axes[i].plot(times/1000, true, "k--", label="FEM"); axes[i].plot(times/1000, pred, "r-", label="Prediction")
        axes[i].set_title(f"{label}: RMSE {row['rmse_degC']:.2f}°C"); axes[i].set_ylabel("T (°C)"); axes[i].grid(alpha=.25)
        if i == 0: axes[i].legend()
        ax_err.plot(times/1000, err, label=label)
    for ax in axes:
        for switch in (19, 39): ax.axvline(switch, color="gray", linestyle=":")
    for ax in axes[-ncols:]:
        if ax.has_data(): ax.set_xlabel("Time (ks)")
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "point_temperature_traces_2col.png"), dpi=dpi); fig.savefig(os.path.join(output_dir, "point_temperature_traces_2col.pdf")); plt.close(fig)
    ax_err.axhline(0, color="black", linewidth=.8)
    for switch in (19, 39): ax_err.axvline(switch, color="gray", linestyle=":")
    ax_err.set(xlabel="Time (ks)", ylabel="Prediction error (°C)", title="Point-wise temperature prediction error"); ax_err.grid(alpha=.3); ax_err.legend(ncol=2)
    fig_err.tight_layout(); fig_err.savefig(os.path.join(output_dir, "point_error_traces_2col.png"), dpi=dpi); fig_err.savefig(os.path.join(output_dir, "point_error_traces_2col.pdf")); plt.close(fig_err)
    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "point_metrics.csv"), index=False)
    with open(os.path.join(output_dir, "point_trace_summary.json"), "w") as f: json.dump({"selected_points": points, "point_metrics": metrics}, f, indent=2)
    return metrics


def main():
    args = parse_args(); os.makedirs(args.output_dir, exist_ok=True); set_plot_style()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 80); print("DIRECT EWMA VARYING-LOAD INFERENCE"); print("=" * 80)
    model, config, meta = load_model_and_meta(args.model_dir, args.dataset, device)
    print(f"Model dir: {args.model_dir}\nDataset: {args.dataset}\nInput channels: {meta['in_channels']}\nEWMA spans: {meta['ewma_spans']}")
    print(f"Model dT mean/std: {meta['dT_mean']:.6f}/{meta['dT_std']:.6f}")
    print(f"Dataset dT mean/std: {meta['dataset_dT_mean']:.6f}/{meta['dataset_dT_std']:.6f}")
    print(f"Normalization source: {meta['normalization_source']}\nDevice: {get_device_info(device)}")

    df = pd.read_csv(args.val_csv); df.columns = df.columns.str.strip(); df = df.dropna()
    first_time = float(df[df.columns[0]].min())
    initial_field, actual = rasterize_csv_at_time(df, first_time, args.coord_file, meta["x_grid"], meta["y_grid"], meta["mask"])
    print(f"Initial GT time: {actual:.1f}s; initial Tmax: {initial_field[meta['mask'] > 0].max():.2f}C")
    times, true_max, true_mean = load_or_create_gt_cache(df, args.coord_file, meta["x_grid"], meta["y_grid"], meta["mask"], args.gt_cache)
    if len(times) == 0: raise ValueError("Ground-truth time array is empty.")

    if device.type == "cuda":
        warm_ewma = np.zeros(
            len(meta["ewma_spans"]),
            dtype=np.float32,
        )

        _ = predict_field(
            model,
            meta,
            initial_field,
            float(times[0]),
            args.t_end,
            warm_ewma,
            device,
        )

        torch.cuda.synchronize(device)

    if device.type == "cuda": torch.cuda.synchronize(device)
    start = time.perf_counter()
    pred_max, pred_mean, _, load_memory = predict_direct_times(model, meta, initial_field, times, args.t_end, device)
    
    validate_first_ewma_state(
        times,
        load_memory,
    )

    if device.type == "cuda": torch.cuda.synchronize(device)
    inference_s = time.perf_counter() - start
    per_point_s = inference_s / len(times)
    print(f"Full-profile inference time: {inference_s:.6f} s")
    print(f"Mean inference time per time point: {per_point_s * 1000:.6f} ms")

    metrics = {
        "T_max_rmse": float(np.sqrt(np.mean((pred_max - true_max)**2))),
        "T_mean_rmse": float(np.sqrt(np.mean((pred_mean - true_mean)**2))),
        "T_max_mae": float(np.mean(np.abs(pred_max - true_max))),
        "T_mean_mae": float(np.mean(np.abs(pred_mean - true_mean))),
        "T_max_max_err": float(np.max(np.abs(pred_max - true_max))),
        "T_mean_max_err": float(np.max(np.abs(pred_mean - true_mean))),
        "normalization_source": meta["normalization_source"],
        "model_dT_mean": meta["dT_mean"], "model_dT_std": meta["dT_std"],
        "dataset_dT_mean": meta["dataset_dT_mean"], "dataset_dT_std": meta["dataset_dT_std"],
        "inference_time_s": inference_s, "inference_time_ms": inference_s * 1000,
        "n_inference_time_points": int(len(times)),
        "inference_time_per_point_s": per_point_s, "inference_time_per_point_ms": per_point_s * 1000,
        "timing_scope": "full-profile neural-operator inference after one CUDA warm-up; excludes model/data loading, FEM rasterization, metrics, plotting, and file output",
        "device_info": get_device_info(device),
        "ewma_spans": meta["ewma_spans"], "model_dir": args.model_dir, "dataset": args.dataset, "gt_cache": args.gt_cache,
        "ewma_alignment": meta["ewma_alignment"],
        "heat_source_mode": meta["heat_source_mode"],
        "Q_input_definition": meta["Q_input_definition"],
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ["T_max_rmse", "T_mean_rmse", "T_max_mae", "T_mean_mae"]}, indent=2))

    save_load_memory(load_memory, args.output_dir, args.dpi)
    plot_prediction_timeseries(times, true_max, true_mean, pred_max, pred_mean, args.output_dir, metrics["T_max_rmse"], metrics["T_mean_rmse"], args.dpi)
    plot_field_comparison(model, meta, df, args.coord_file, initial_field, times, args.output_dir, device, args.t_end, args.dpi)
    points = select_monitoring_points(df, args.coord_file, meta["x_grid"], meta["y_grid"], meta["mask"])
    plot_selected_monitoring_points(df, args.coord_file, meta, points, args.output_dir, args.dpi)
    point_times, true_traces = get_true_point_traces(df, points)
    pred_traces = predict_point_traces(model, meta, initial_field, point_times, points, args.t_end, device)
    plot_point_outputs(df, args.coord_file, meta, points, point_times, true_traces, pred_traces, args.output_dir, args.dpi)
    print(f"\nDone. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
