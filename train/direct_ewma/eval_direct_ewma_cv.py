"""
eval_direct_ewma_cv.py

Evaluate leave-one-condition-out CV folds for the direct EWMA FNO model.

Model input:
    [u0, t_norm, x, y, Q_base, p, ewma_3, ewma_10, ewma_30, ...]

Model output:
    dT(t) = T(t) - u0

This script:
    1. Finds CV fold folders under --cv_root.
    2. Loads each fold model and test_indices.npy.
    3. Predicts held-out constant-load condition directly.
    4. Computes:
        - field RMSE / MAE
        - T_max RMSE / MAE
        - T_mean RMSE / MAE
    5. Saves:
        - per-fold metrics
        - time-series plots
        - field comparison examples
        - point-wise traces
        - global cv_summary.csv/json

Usage:
    python eval_direct_ewma_cv.py \
        --cv_root results/direct_ewma_cv_4cond \
        --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 \
        --device cuda:1 \
        --model_file final_model.pth
"""

import argparse
import glob
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from neuralop.models import FNO


# ============================================================
# Args
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cv_root", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/fno_6cond_element_512_direct_ewma.h5",
    )
    parser.add_argument("--model_file", type=str, default="final_model.pth")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--fold_glob", type=str, default="test_*")
    parser.add_argument("--output_prefix", type=str, default="cv_summary")

    parser.add_argument("--n_field_examples", type=int, default=6)
    parser.add_argument("--n_points", type=int, default=6)
    parser.add_argument("--drop_first_n_points", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=600)

    parser.add_argument("--t_end", type=float, default=60000.0)

    return parser.parse_args()


# ============================================================
# Style / utils
# ============================================================
def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def condition_label(cond):
    mapping = {
        "25": "25%",
        "half": "50%",
        "50": "50%",
        "75": "75%",
        "full": "100%",
        "100": "100%",
        "62.5": "62.5%",
        "87.5": "87.5%",
    }
    return mapping.get(str(cond), str(cond))


def condition_sort_key(cond):
    order = {
        "25": 0,
        "half": 1,
        "50": 1,
        "75": 2,
        "full": 3,
        "100": 3,
        "62.5": 4,
        "87.5": 5,
    }
    return order.get(str(cond), 999)


def infer_condition_from_folder(folder_name):
    name = folder_name.replace("test_", "")
    mapping = {
        "25": "25",
        "50": "half",
        "half": "half",
        "75": "75",
        "100": "full",
        "full": "full",
    }
    return mapping.get(name, name)


# ============================================================
# Loading
# ============================================================
def load_config(fold_dir):
    config_path = os.path.join(fold_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


# def load_dataset(dataset_path, device):
#     with h5py.File(dataset_path, "r") as f:
#         X_all = torch.tensor(f["X_all"][:], dtype=torch.float32)
#         Y_all = torch.tensor(f["Y_all"][:], dtype=torch.float32)

#         mask_np = f["mask"][:].astype(np.float32)
#         mask_torch = torch.tensor(mask_np, dtype=torch.float32, device=device)

#         x_grid = f["x_grid"][:]
#         y_grid = f["y_grid"][:]

#         T_abs_mean = float(f.attrs["T_abs_mean"])
#         T_abs_std = float(f.attrs["T_abs_std"])

#         # Direct EWMA target normalization
#         if "dT_mean" in f.attrs and "dT_std" in f.attrs:
#             dT_mean = float(config["dT_mean"])
#             dT_std = float(config["dT_std"])
#         else:
#             raise KeyError("Dataset must contain attrs dT_mean and dT_std")

#         in_channels = int(f["X_all"].shape[1])
#         ewma_spans = json.loads(f.attrs["ewma_spans_json"]) if "ewma_spans_json" in f.attrs else []

#     meta = {
#         "mask_np": mask_np,
#         "mask_torch": mask_torch,
#         "x_grid": x_grid,
#         "y_grid": y_grid,
#         "T_abs_mean": T_abs_mean,
#         "T_abs_std": T_abs_std,
#         "dT_mean": dT_mean,
#         "dT_std": dT_std,
#         "in_channels": in_channels,
#         "ewma_spans": ewma_spans,
#     }

#     return X_all, Y_all, meta
def load_dataset(dataset_path, device):
    """
    Load the complete dataset.

    Important:
        Y_all is stored using dataset-global normalization.
        Therefore, the normalization values loaded here are the
        original HDF5 dataset statistics.

        Fold-specific model-output normalization is loaded separately
        from each fold's config.json.
    """
    with h5py.File(dataset_path, "r") as f:
        X_all = torch.tensor(
            f["X_all"][:],
            dtype=torch.float32,
        )

        Y_all = torch.tensor(
            f["Y_all"][:],
            dtype=torch.float32,
        )

        mask_np = f["mask"][:].astype(np.float32)

        mask_torch = torch.tensor(
            mask_np,
            dtype=torch.float32,
            device=device,
        )

        x_grid = f["x_grid"][:]
        y_grid = f["y_grid"][:]

        T_abs_mean = float(f.attrs["T_abs_mean"])
        T_abs_std = float(f.attrs["T_abs_std"])

        if "dT_mean" not in f.attrs or "dT_std" not in f.attrs:
            raise KeyError(
                "Dataset must contain dT_mean and dT_std attributes."
            )

        # Original normalization used to store Y_all in the HDF5 file.
        dataset_dT_mean = float(f.attrs["dT_mean"])
        dataset_dT_std = float(f.attrs["dT_std"])

        in_channels = int(f["X_all"].shape[1])

        if "ewma_spans_json" in f.attrs:
            ewma_spans = json.loads(
                f.attrs["ewma_spans_json"]
            )
        else:
            ewma_spans = []

    meta = {
        "mask_np": mask_np,
        "mask_torch": mask_torch,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "T_abs_mean": T_abs_mean,
        "T_abs_std": T_abs_std,

        # These statistics describe Y_all in the HDF5 file.
        "dataset_dT_mean": dataset_dT_mean,
        "dataset_dT_std": dataset_dT_std,

        "in_channels": in_channels,
        "ewma_spans": ewma_spans,
    }

    return X_all, Y_all, meta


def load_model(fold_dir, config, in_channels, device, model_file="best_model.pth"):
    model_path = os.path.join(fold_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = FNO(
        n_modes=(config["modes"], config["modes"]),
        in_channels=in_channels,
        out_channels=1,
        hidden_channels=config["width"],
        n_layers=config["n_layers"],
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# Prediction / metrics
# ============================================================
@torch.no_grad()
def predict_all(model, X, batch_size):
    preds = []

    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        pred = model(xb)
        preds.append(pred.detach().cpu())

    return torch.cat(preds, dim=0)


# def denormalize_fields(X_cpu, Y_norm_cpu, pred_norm_cpu, meta):
#     """
#     Returns:
#         true_T: [N,H,W]
#         pred_T: [N,H,W]
#         time_s: [N]
#         p_values: [N]
#     """
#     X_np = X_cpu.numpy()
#     Y_np = Y_norm_cpu.numpy()
#     P_np = pred_norm_cpu.numpy()

#     mask = meta["mask_np"]

#     u0 = X_np[:, 0] * meta["T_abs_std"] + meta["T_abs_mean"]

#     true_dT = Y_np[:, 0] * meta["dT_std"] + meta["dT_mean"]
#     pred_dT = P_np[:, 0] * meta["dT_std"] + meta["dT_mean"]

#     true_T = u0 + true_dT
#     pred_T = u0 + pred_dT

#     # t_norm channel is X[:,1], spatially constant
#     mask_bool = mask > 0
#     time_s = []
#     p_values = []

#     for i in range(len(X_np)):
#         time_s.append(float(np.mean(X_np[i, 1][mask_bool]) * 21780.0))

#         # p channel index = 5 for direct EWMA
#         if X_np.shape[1] > 5:
#             p_values.append(float(np.mean(X_np[i, 5][mask_bool])))
#         else:
#             p_values.append(np.nan)

#     return true_T, pred_T, np.array(time_s), np.array(p_values)
def denormalize_fields(
    X_cpu,
    Y_dataset_norm_cpu,
    pred_fold_norm_cpu,
    meta,
    fold_dT_mean,
    fold_dT_std,
):
    """
    Convert the ground-truth and predicted temperature-rise fields
    back to physical units.

    Normalization sources:
        Ground truth Y:
            Stored in HDF5 using dataset-global normalization.

        Model prediction:
            Produced using fold-specific normalization fitted from
            that fold's training split.

    Returns:
        true_T:  [N, H, W]
        pred_T:  [N, H, W]
        time_s:  [N]
        p_values: [N]
    """
    X_np = X_cpu.numpy()
    Y_np = Y_dataset_norm_cpu.numpy()
    pred_np = pred_fold_norm_cpu.numpy()

    mask = meta["mask_np"]
    mask_bool = mask > 0

    # Initial absolute temperature.
    # This keeps the existing input-channel normalization.
    u0 = (
        X_np[:, 0] * meta["T_abs_std"]
        + meta["T_abs_mean"]
    )

    # Ground truth is stored using the original HDF5 statistics.
    true_dT = (
        Y_np[:, 0] * meta["dataset_dT_std"]
        + meta["dataset_dT_mean"]
    )

    # Prediction uses this fold's training-derived statistics.
    pred_dT = (
        pred_np[:, 0] * fold_dT_std
        + fold_dT_mean
    )

    true_T = u0 + true_dT
    pred_T = u0 + pred_dT

    # Prevent values outside the geometry from affecting displays.
    true_T = true_T * mask[None, :, :]
    pred_T = pred_T * mask[None, :, :]

    # Constant-load physical time:
    # t_norm in [0,1] corresponds to 0-21780 s.
    time_s = []
    p_values = []

    for i in range(len(X_np)):
        t_norm_i = float(
            np.mean(X_np[i, 1][mask_bool])
        )
        time_s.append(t_norm_i * 21780.0)

        # Current load p is input channel 5.
        if X_np.shape[1] > 5:
            p_values.append(
                float(np.mean(X_np[i, 5][mask_bool]))
            )
        else:
            p_values.append(np.nan)

    return (
        true_T,
        pred_T,
        np.asarray(time_s),
        np.asarray(p_values),
    )


def compute_metrics(true_T, pred_T, mask):
    mask_bool = mask > 0

    err = pred_T - true_T

    field_rmse_each = []
    field_mae_each = []

    true_Tmax = []
    pred_Tmax = []
    true_Tmean = []
    pred_Tmean = []

    for i in range(len(true_T)):
        e = err[i][mask_bool]
        tv = true_T[i][mask_bool]
        pv = pred_T[i][mask_bool]

        field_rmse_each.append(float(np.sqrt(np.mean(e ** 2))))
        field_mae_each.append(float(np.mean(np.abs(e))))

        true_Tmax.append(float(np.max(tv)))
        pred_Tmax.append(float(np.max(pv)))
        true_Tmean.append(float(np.mean(tv)))
        pred_Tmean.append(float(np.mean(pv)))

    true_Tmax = np.array(true_Tmax)
    pred_Tmax = np.array(pred_Tmax)
    true_Tmean = np.array(true_Tmean)
    pred_Tmean = np.array(pred_Tmean)

    metrics = {
        "field_rmse_avg_degC": float(np.mean(field_rmse_each)),
        "field_mae_avg_degC": float(np.mean(field_mae_each)),
        "field_rmse_max_degC": float(np.max(field_rmse_each)),
        "field_mae_max_degC": float(np.max(field_mae_each)),

        "Tmax_rmse_degC": float(np.sqrt(np.mean((pred_Tmax - true_Tmax) ** 2))),
        "Tmax_mae_degC": float(np.mean(np.abs(pred_Tmax - true_Tmax))),
        "Tmax_max_err_degC": float(np.max(np.abs(pred_Tmax - true_Tmax))),

        "Tmean_rmse_degC": float(np.sqrt(np.mean((pred_Tmean - true_Tmean) ** 2))),
        "Tmean_mae_degC": float(np.mean(np.abs(pred_Tmean - true_Tmean))),
        "Tmean_max_err_degC": float(np.max(np.abs(pred_Tmean - true_Tmean))),
    }

    series = {
        "true_Tmax": true_Tmax,
        "pred_Tmax": pred_Tmax,
        "true_Tmean": true_Tmean,
        "pred_Tmean": pred_Tmean,
        "field_rmse_each": np.array(field_rmse_each),
        "field_mae_each": np.array(field_mae_each),
    }

    return metrics, series


# ============================================================
# Plotting
# ============================================================
def plot_timeseries(time_s, series, test_condition, fold_dir, dpi):
    output_dir = os.path.join(fold_dir, "visualization")
    os.makedirs(output_dir, exist_ok=True)

    t_ks = time_s / 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(7.16, 6.0), sharex=True)

    axes[0].plot(t_ks, series["true_Tmax"], "k--", label="FEM $T_{max}$")
    axes[0].plot(t_ks, series["pred_Tmax"], "r-", label="Pred. $T_{max}$")
    axes[0].set_ylabel("$T_{max}$ (°C)")
    axes[0].set_title(f"Direct EWMA CV: held-out {condition_label(test_condition)}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_ks, series["true_Tmean"], "k--", label="FEM $T_{mean}$")
    axes[1].plot(t_ks, series["pred_Tmean"], "b-", label="Pred. $T_{mean}$")
    axes[1].set_ylabel("$T_{mean}$ (°C)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_ks, series["pred_Tmax"] - series["true_Tmax"], "r-", label="$T_{max}$ error")
    axes[2].plot(t_ks, series["pred_Tmean"] - series["true_Tmean"], "b-", label="$T_{mean}$ error")
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set_ylabel("Error (°C)")
    axes[2].set_xlabel("Time (ks)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"timeseries_Tmax_Tmean.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


# def plot_field_examples(true_T, pred_T, time_s, meta, test_condition, fold_dir, n_examples, dpi):
#     output_dir = os.path.join(fold_dir, "visualization")
#     os.makedirs(output_dir, exist_ok=True)

#     mask = meta["mask_np"]
#     mask_bool = mask > 0
#     x_grid = meta["x_grid"]
#     y_grid = meta["y_grid"]
#     extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]

#     n = len(true_T)
#     n_examples = min(n_examples, n)
#     example_ids = np.linspace(0, n - 1, n_examples, dtype=int)

#     err_T = pred_T - true_T

#     vmin_T = min(pred_T[:, mask_bool].min(), true_T[:, mask_bool].min())
#     vmax_T = max(pred_T[:, mask_bool].max(), true_T[:, mask_bool].max())
#     err_lim = max(1.0, np.percentile(np.abs(err_T[:, mask_bool]), 99))

#     fig, axes = plt.subplots(3, n_examples, figsize=(7.16, 4.2))

#     if n_examples == 1:
#         axes = np.expand_dims(axes, axis=1)

#     im_temp = None
#     im_err = None

#     for col, idx in enumerate(example_ids):
#         pred_disp = np.where(mask_bool, pred_T[idx], np.nan)
#         true_disp = np.where(mask_bool, true_T[idx], np.nan)
#         err_disp = np.where(mask_bool, err_T[idx], np.nan)

#         im_temp = axes[0, col].imshow(
#             pred_disp,
#             origin="lower",
#             extent=extent,
#             cmap="jet",
#             vmin=vmin_T,
#             vmax=vmax_T,
#             aspect="equal",
#         )
#         axes[0, col].set_title(f"t={time_s[idx]/1000:.1f} ks")
#         axes[0, col].axis("off")

#         axes[1, col].imshow(
#             true_disp,
#             origin="lower",
#             extent=extent,
#             cmap="jet",
#             vmin=vmin_T,
#             vmax=vmax_T,
#             aspect="equal",
#         )
#         axes[1, col].axis("off")

#         im_err = axes[2, col].imshow(
#             err_disp,
#             origin="lower",
#             extent=extent,
#             cmap="RdBu_r",
#             vmin=-err_lim,
#             vmax=err_lim,
#             aspect="equal",
#         )
#         mae = float(np.mean(np.abs(err_T[idx][mask_bool])))
#         axes[2, col].set_title(f"MAE={mae:.2f}°C")
#         axes[2, col].axis("off")

#     axes[0, 0].set_ylabel("Prediction", fontsize=9, fontweight="bold")
#     axes[1, 0].set_ylabel("FEM", fontsize=9, fontweight="bold")
#     axes[2, 0].set_ylabel("Error", fontsize=9, fontweight="bold")

#     fig.suptitle(f"Field comparison, held-out {condition_label(test_condition)}", y=0.98)

#     fig.subplots_adjust(right=0.90, hspace=0.12, wspace=0.04)

#     cbar_ax1 = fig.add_axes([0.91, 0.40, 0.012, 0.47])
#     fig.colorbar(im_temp, cax=cbar_ax1, label="Temperature (°C)")

#     cbar_ax2 = fig.add_axes([0.91, 0.10, 0.012, 0.22])
#     fig.colorbar(im_err, cax=cbar_ax2, label="Error (°C)")

#     for ext in ["png", "pdf"]:
#         path = os.path.join(output_dir, f"field_examples.{ext}")
#         fig.savefig(path, dpi=dpi, bbox_inches="tight")

#     plt.close(fig)
def plot_field_examples(
    true_T,
    pred_T,
    time_s,
    meta,
    test_condition,
    fold_dir,
    n_examples,
    dpi,
):
    output_dir = os.path.join(fold_dir, "visualization")
    os.makedirs(output_dir, exist_ok=True)

    mask = meta["mask_np"]
    mask_bool = mask > 0
    x_grid = meta["x_grid"]
    y_grid = meta["y_grid"]
    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]

    n = len(true_T)
    n_examples = min(n_examples, n)
    example_ids = np.linspace(0, n - 1, n_examples, dtype=int)

    err_T = pred_T - true_T

    vmin_T = min(
        pred_T[:, mask_bool].min(),
        true_T[:, mask_bool].min(),
    )
    vmax_T = max(
        pred_T[:, mask_bool].max(),
        true_T[:, mask_bool].max(),
    )

    # Transparent scientific scaling: include every error value.
    err_lim = max(
        1.0,
        float(np.max(np.abs(err_T[:, mask_bool]))),
    )

    # Generate directly at IEEE two-column width.
    fig, axes = plt.subplots(
        3,
        n_examples,
        figsize=(7.16, 4.2),
    )

    if n_examples == 1:
        axes = np.expand_dims(axes, axis=1)

    im_temp = None
    im_err = None

    for col, idx in enumerate(example_ids):
        pred_disp = np.where(mask_bool, pred_T[idx], np.nan)
        true_disp = np.where(mask_bool, true_T[idx], np.nan)
        err_disp = np.where(mask_bool, err_T[idx], np.nan)

        im_temp = axes[0, col].imshow(
            pred_disp,
            origin="lower",
            extent=extent,
            cmap="jet",
            vmin=vmin_T,
            vmax=vmax_T,
            aspect="equal",
        )
        axes[0, col].set_title(
            f"$t={time_s[idx] / 1000:.1f}$ ks"
        )

        axes[1, col].imshow(
            true_disp,
            origin="lower",
            extent=extent,
            cmap="jet",
            vmin=vmin_T,
            vmax=vmax_T,
            aspect="equal",
        )

        im_err = axes[2, col].imshow(
            err_disp,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-err_lim,
            vmax=err_lim,
            aspect="equal",
        )

        mae = float(np.mean(np.abs(err_T[idx][mask_bool])))
        axes[2, col].set_title(
            f"MAE = {mae:.2f} $^\\circ$C"
        )

        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    # Row labels independent of hidden axes.
    fig.text(
        0.015,
        0.765,
        "Prediction",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
    fig.text(
        0.015,
        0.495,
        "FEM",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
    fig.text(
        0.015,
        0.225,
        "Error",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

    fig.suptitle(
        f"Field comparison, held-out "
        f"{condition_label(test_condition)}",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.90,
        top=0.90,
        bottom=0.05,
        hspace=0.28,
        wspace=0.04,
    )

    # Temperature colorbar.
    cbar_ax1 = fig.add_axes([0.915, 0.39, 0.012, 0.48])
    cbar1 = fig.colorbar(im_temp, cax=cbar_ax1)
    cbar1.set_label("Temperature ($^\\circ$C)", fontsize=9)
    cbar1.ax.tick_params(labelsize=9)

    # Error colorbar.
    cbar_ax2 = fig.add_axes([0.915, 0.08, 0.012, 0.22])
    cbar2 = fig.colorbar(im_err, cax=cbar_ax2)
    cbar2.set_label("Error ($^\\circ$C)", fontsize=9)
    cbar2.ax.tick_params(labelsize=9)

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"field_examples.{ext}")
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )
        print(f"Saved: {path}")

    plt.close(fig)


# ============================================================
# Point traces
# ============================================================
def select_points_from_true_field(true_T, meta, n_points=6, drop_first_n=2):
    """
    Select points from final true field based on temperature quantiles.
    Then optionally drop lowest-temp points.
    """
    mask = meta["mask_np"]
    mask_bool = mask > 0

    final_field = true_T[-1]
    iy_all, ix_all = np.where(mask_bool)
    temps = final_field[iy_all, ix_all]

    quantiles = np.linspace(0.05, 0.95, n_points)
    selected = []

    used = set()

    for q in quantiles:
        target = np.quantile(temps, q)
        order = np.argsort(np.abs(temps - target))

        chosen = None
        for idx in order:
            key = (int(ix_all[idx]), int(iy_all[idx]))
            if key not in used:
                chosen = idx
                break

        if chosen is None:
            chosen = order[0]

        ix = int(ix_all[chosen])
        iy = int(iy_all[chosen])
        used.add((ix, iy))

        selected.append({
            "label": f"P{len(selected)+1}",
            "ix": ix,
            "iy": iy,
            "x": float(meta["x_grid"][ix]),
            "y": float(meta["y_grid"][iy]),
            "T_ref": float(final_field[iy, ix]),
        })

    if drop_first_n > 0:
        selected = selected[drop_first_n:]

    for i, p in enumerate(selected, start=1):
        p["label"] = f"P{i}"

    return selected


def plot_point_traces(true_T, pred_T, time_s, selected_points, test_condition, fold_dir, dpi):
    output_dir = os.path.join(fold_dir, "visualization")
    os.makedirs(output_dir, exist_ok=True)

    n = len(selected_points)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, 1.85 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    t_ks = time_s / 1000.0
    metrics_rows = []

    for idx, p in enumerate(selected_points):
        ax = axes[idx]
        label = p["label"]
        ix, iy = p["ix"], p["iy"]

        true_vals = true_T[:, iy, ix]
        pred_vals = pred_T[:, iy, ix]
        err = pred_vals - true_vals

        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        max_err = float(np.max(np.abs(err)))

        metrics_rows.append({
            "label": label,
            "ix": ix,
            "iy": iy,
            "x": p["x"],
            "y": p["y"],
            "rmse_degC": rmse,
            "mae_degC": mae,
            "max_err_degC": max_err,
        })

        ax.plot(t_ks, true_vals, "k--", label="FEM")
        ax.plot(t_ks, pred_vals, "r-", label="Prediction")
        ax.set_title(f"{label}: RMSE {rmse:.2f}°C")
        ax.set_ylabel("T (°C)")
        ax.grid(alpha=0.3)

        if idx == 0:
            ax.legend(loc="best")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    for idx, ax in enumerate(axes):
        if idx >= (nrows - 1) * ncols and ax.has_data():
            ax.set_xlabel("Time (ks)")

    fig.suptitle(f"Point-wise temperature traces, held-out {condition_label(test_condition)}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"point_temperature_traces.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)

    # Error plot
    fig, ax = plt.subplots(figsize=(7.16, 3.2))

    for p in selected_points:
        ix, iy = p["ix"], p["iy"]
        err = pred_T[:, iy, ix] - true_T[:, iy, ix]
        ax.plot(t_ks, err, label=p["label"])

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Time (ks)")
    ax.set_ylabel("Prediction error (°C)")
    ax.set_title(f"Point-wise error, held-out {condition_label(test_condition)}")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"point_error_traces.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)

    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(output_dir, "point_metrics.csv"),
        index=False,
    )

    return metrics_rows


# ============================================================
# Per fold evaluation
# ============================================================
# def evaluate_fold(fold_dir, args, X_all, Y_all, meta, device):
#     fold_name = os.path.basename(fold_dir)

#     config = load_config(fold_dir)
#     test_idx_path = os.path.join(fold_dir, "test_indices.npy")

#     if not os.path.exists(test_idx_path):
#         raise FileNotFoundError(f"Missing test_indices.npy: {test_idx_path}")

#     test_idx = np.load(test_idx_path)

#     test_condition = config.get("test_condition", None)
#     if test_condition is None:
#         test_condition = infer_condition_from_folder(fold_name)

#     model = load_model(
#         fold_dir=fold_dir,
#         config=config,
#         in_channels=meta["in_channels"],
#         device=device,
#         model_file=args.model_file,
#     )

#     X_test = X_all[test_idx].to(device)
#     Y_test = Y_all[test_idx].to(device)

#     print(f"\n{'='*90}")
#     print(f"Fold: {fold_name}")
#     print(f"Test condition: {test_condition} ({condition_label(test_condition)})")
#     print(f"X_test: {tuple(X_test.shape)}")
#     print(f"Y_test: {tuple(Y_test.shape)}")

#     pred_norm = predict_all(model, X_test, args.batch_size)

#     X_cpu = X_test.detach().cpu()
#     Y_cpu = Y_test.detach().cpu()

#     true_T, pred_T, time_s, p_values = denormalize_fields(
#         X_cpu=X_cpu,
#         Y_norm_cpu=Y_cpu,
#         pred_norm_cpu=pred_norm,
#         meta=meta,
#     )

#     metrics, series = compute_metrics(true_T, pred_T, meta["mask_np"])

#     print(
#         f"Tmax RMSE={metrics['Tmax_rmse_degC']:.4f}°C | "
#         f"Tmean RMSE={metrics['Tmean_rmse_degC']:.4f}°C | "
#         f"Field MAE={metrics['field_mae_avg_degC']:.4f}°C"
#     )

#     # save per-fold metrics
#     row = {
#         "fold": fold_name,
#         "test_condition": test_condition,
#         "test_condition_label": condition_label(test_condition),
#         "n_test_samples": int(len(test_idx)),
#         "model_file_used": args.model_file,
#         **metrics,
#     }

#     metrics_path = os.path.join(fold_dir, "cv_inference_metrics.json")
#     with open(metrics_path, "w") as f:
#         json.dump(row, f, indent=2)

#     # plots
#     plot_timeseries(time_s, series, test_condition, fold_dir, args.dpi)
#     plot_field_examples(true_T, pred_T, time_s, meta, test_condition, fold_dir, args.n_field_examples, args.dpi)

#     selected_points = select_points_from_true_field(
#         true_T=true_T,
#         meta=meta,
#         n_points=args.n_points,
#         drop_first_n=args.drop_first_n_points,
#     )

#     point_metrics = plot_point_traces(
#         true_T=true_T,
#         pred_T=pred_T,
#         time_s=time_s,
#         selected_points=selected_points,
#         test_condition=test_condition,
#         fold_dir=fold_dir,
#         dpi=args.dpi,
#     )

#     row["point_metrics"] = point_metrics

#     with open(metrics_path, "w") as f:
#         json.dump(row, f, indent=2)

#     del model, X_test, Y_test
#     if device.type == "cuda":
#         torch.cuda.empty_cache()

#     return row
def evaluate_fold(
    fold_dir,
    args,
    X_all,
    Y_all,
    meta,
    device,
):
    fold_name = os.path.basename(fold_dir)

    # --------------------------------------------------------
    # Load this fold's training configuration
    # --------------------------------------------------------
    config = load_config(fold_dir)

    if "dT_mean" not in config or "dT_std" not in config:
        raise KeyError(
            f"Fold configuration does not contain fold-specific "
            f"dT_mean/dT_std: {fold_dir}/config.json"
        )

    fold_dT_mean = float(config["dT_mean"])
    fold_dT_std = float(config["dT_std"])

    normalization_info = config.get(
        "normalization",
        {},
    )

    normalization_source = normalization_info.get(
        "source",
        "unknown",
    )

    if normalization_source != (
        "training_samples_and_active_pixels_only"
    ):
        print(
            "WARNING: This fold may not use the clean "
            "training-only normalization protocol."
        )
        print(
            f"Normalization source: {normalization_source}"
        )

    # --------------------------------------------------------
    # Load the exact held-out indices saved during training
    # --------------------------------------------------------
    test_idx_path = os.path.join(
        fold_dir,
        "test_indices.npy",
    )

    if not os.path.exists(test_idx_path):
        raise FileNotFoundError(
            f"Missing test_indices.npy: {test_idx_path}"
        )

    test_idx = np.load(test_idx_path)

    if len(test_idx) == 0:
        raise ValueError(
            f"Test index file is empty: {test_idx_path}"
        )

    # --------------------------------------------------------
    # Determine held-out operating condition
    # --------------------------------------------------------
    test_condition = config.get(
        "test_condition",
        None,
    )

    if test_condition is None:
        test_condition = infer_condition_from_folder(
            fold_name
        )

    # --------------------------------------------------------
    # Load validation-selected fold model
    # --------------------------------------------------------
    model = load_model(
        fold_dir=fold_dir,
        config=config,
        in_channels=meta["in_channels"],
        device=device,
        model_file=args.model_file,
    )

    X_test = X_all[test_idx].to(device)

    # Keep Y_test in its original HDF5 normalization.
    # denormalize_fields() handles the two normalization systems.
    Y_test_dataset_norm = Y_all[test_idx]

    print(f"\n{'=' * 90}")
    print(f"Fold: {fold_name}")
    print(
        f"Test condition: {test_condition} "
        f"({condition_label(test_condition)})"
    )
    print(f"Model file: {args.model_file}")
    print(
        f"Dataset-global dT mean/std: "
        f"{meta['dataset_dT_mean']:.6f}/"
        f"{meta['dataset_dT_std']:.6f}"
    )
    print(
        f"Fold-specific dT mean/std: "
        f"{fold_dT_mean:.6f}/"
        f"{fold_dT_std:.6f}"
    )
    print(
        f"Normalization source: {normalization_source}"
    )
    print(f"X_test: {tuple(X_test.shape)}")
    print(
        f"Y_test: {tuple(Y_test_dataset_norm.shape)}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    pred_fold_norm = predict_all(
        model=model,
        X=X_test,
        batch_size=args.batch_size,
    )

    X_cpu = X_test.detach().cpu()

    # Y_test is already on CPU because Y_all is loaded on CPU.
    Y_cpu = Y_test_dataset_norm.detach().cpu()

    # --------------------------------------------------------
    # Convert ground truth and prediction to physical units
    # --------------------------------------------------------
    true_T, pred_T, time_s, p_values = denormalize_fields(
        X_cpu=X_cpu,
        Y_dataset_norm_cpu=Y_cpu,
        pred_fold_norm_cpu=pred_fold_norm,
        meta=meta,
        fold_dT_mean=fold_dT_mean,
        fold_dT_std=fold_dT_std,
    )

    # Sort by physical time before calculating traces and plotting.
    order = np.argsort(time_s)

    time_s = time_s[order]
    p_values = p_values[order]
    true_T = true_T[order]
    pred_T = pred_T[order]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    metrics, series = compute_metrics(
        true_T=true_T,
        pred_T=pred_T,
        mask=meta["mask_np"],
    )

    print(
        f"Tmax RMSE="
        f"{metrics['Tmax_rmse_degC']:.4f} C | "
        f"Tmean RMSE="
        f"{metrics['Tmean_rmse_degC']:.4f} C | "
        f"Field MAE="
        f"{metrics['field_mae_avg_degC']:.4f} C"
    )

    # --------------------------------------------------------
    # Save per-fold metrics
    # --------------------------------------------------------
    row = {
        "fold": fold_name,
        "test_condition": test_condition,
        "test_condition_label": condition_label(
            test_condition
        ),
        "n_test_samples": int(len(test_idx)),
        "model_file_used": args.model_file,

        "normalization_source": normalization_source,
        "dataset_global_dT_mean": float(
            meta["dataset_dT_mean"]
        ),
        "dataset_global_dT_std": float(
            meta["dataset_dT_std"]
        ),
        "fold_dT_mean": fold_dT_mean,
        "fold_dT_std": fold_dT_std,

        **metrics,
    }

    metrics_path = os.path.join(
        fold_dir,
        "cv_inference_metrics.json",
    )

    with open(metrics_path, "w") as file:
        json.dump(row, file, indent=2)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plot_timeseries(
        time_s=time_s,
        series=series,
        test_condition=test_condition,
        fold_dir=fold_dir,
        dpi=args.dpi,
    )

    plot_field_examples(
        true_T=true_T,
        pred_T=pred_T,
        time_s=time_s,
        meta=meta,
        test_condition=test_condition,
        fold_dir=fold_dir,
        n_examples=args.n_field_examples,
        dpi=args.dpi,
    )

    selected_points = select_points_from_true_field(
        true_T=true_T,
        meta=meta,
        n_points=args.n_points,
        drop_first_n=args.drop_first_n_points,
    )

    point_metrics = plot_point_traces(
        true_T=true_T,
        pred_T=pred_T,
        time_s=time_s,
        selected_points=selected_points,
        test_condition=test_condition,
        fold_dir=fold_dir,
        dpi=args.dpi,
    )

    row["point_metrics"] = point_metrics

    with open(metrics_path, "w") as file:
        json.dump(row, file, indent=2)

    print(f"Saved fold metrics: {metrics_path}")

    del model
    del X_test
    del pred_fold_norm

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return row


# ============================================================
# Global summary
# ============================================================
def save_global_summary(rows, cv_root, output_prefix, dpi):
    df_rows = []

    for r in rows:
        df_rows.append({
            "fold": r["fold"],
            "test_condition": r["test_condition"],
            "test_condition_label": r["test_condition_label"],
            "n_test_samples": r["n_test_samples"],
            "field_rmse_avg_degC": r["field_rmse_avg_degC"],
            "field_mae_avg_degC": r["field_mae_avg_degC"],
            "Tmax_rmse_degC": r["Tmax_rmse_degC"],
            "Tmax_mae_degC": r["Tmax_mae_degC"],
            "Tmean_rmse_degC": r["Tmean_rmse_degC"],
            "Tmean_mae_degC": r["Tmean_mae_degC"],
        })

    df = pd.DataFrame(df_rows)
    df["sort_key"] = df["test_condition"].map(condition_sort_key)
    df = df.sort_values("sort_key").drop(columns=["sort_key"])

    csv_path = os.path.join(cv_root, f"{output_prefix}.csv")
    json_path = os.path.join(cv_root, f"{output_prefix}.json")

    df.to_csv(csv_path, index=False)

    summary = {
        "cv_root": cv_root,
        "n_folds": int(len(df)),
        "field_rmse_mean_degC": float(df["field_rmse_avg_degC"].mean()),
        "field_rmse_std_degC": float(df["field_rmse_avg_degC"].std(ddof=1)) if len(df) > 1 else 0.0,
        "field_mae_mean_degC": float(df["field_mae_avg_degC"].mean()),
        "field_mae_std_degC": float(df["field_mae_avg_degC"].std(ddof=1)) if len(df) > 1 else 0.0,
        "Tmax_rmse_mean_degC": float(df["Tmax_rmse_degC"].mean()),
        "Tmax_rmse_std_degC": float(df["Tmax_rmse_degC"].std(ddof=1)) if len(df) > 1 else 0.0,
        "Tmean_rmse_mean_degC": float(df["Tmean_rmse_degC"].mean()),
        "Tmean_rmse_std_degC": float(df["Tmean_rmse_degC"].std(ddof=1)) if len(df) > 1 else 0.0,
        "folds": df_rows,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print("CV SUMMARY")
    print("=" * 90)
    print(df.to_string(index=False))
    print("-" * 90)
    print(f"Field RMSE mean ± std: {summary['field_rmse_mean_degC']:.4f} ± {summary['field_rmse_std_degC']:.4f} °C")
    print(f"Tmax  RMSE mean ± std: {summary['Tmax_rmse_mean_degC']:.4f} ± {summary['Tmax_rmse_std_degC']:.4f} °C")
    print(f"Tmean RMSE mean ± std: {summary['Tmean_rmse_mean_degC']:.4f} ± {summary['Tmean_rmse_std_degC']:.4f} °C")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")

    # bar plot
    labels = df["test_condition_label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.16, 3.4))

    ax.bar(x - width/2, df["Tmax_rmse_degC"], width, label="$T_{max}$ RMSE")
    ax.bar(x + width/2, df["Tmean_rmse_degC"], width, label="$T_{mean}$ RMSE")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (°C)")
    ax.set_xlabel("Held-out load condition")
    ax.set_title("Direct EWMA FNO leave-one-condition-out CV")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(cv_root, f"{output_prefix}_barplot.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    set_plot_style()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    fold_dirs = sorted(glob.glob(os.path.join(args.cv_root, args.fold_glob)))
    fold_dirs = [
        d for d in fold_dirs
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json"))
    ]

    if len(fold_dirs) == 0:
        raise FileNotFoundError(f"No fold folders found under {args.cv_root}/{args.fold_glob}")

    print("=" * 90)
    print("DIRECT EWMA FNO CV INFERENCE")
    print("=" * 90)
    print(f"CV root:   {args.cv_root}")
    print(f"Dataset:   {args.dataset}")
    print(f"Folds:     {len(fold_dirs)}")
    print(f"Device:    {device}")
    print()

    X_all, Y_all, meta = load_dataset(args.dataset, device)

    print(
        "Dataset-global dT mean/std: "
        f"{meta['dataset_dT_mean']:.6f}/"
        f"{meta['dataset_dT_std']:.6f}"
    )

    print(f"X_all: {tuple(X_all.shape)}")
    print(f"Y_all: {tuple(Y_all.shape)}")
    print(f"Input channels: {meta['in_channels']}")
    print(f"EWMA spans: {meta['ewma_spans']}")
    print()

    rows = []

    for fold_dir in fold_dirs:
        row = evaluate_fold(
            fold_dir=fold_dir,
            args=args,
            X_all=X_all,
            Y_all=Y_all,
            meta=meta,
            device=device,
        )
        rows.append(row)

    save_global_summary(
        rows=rows,
        cv_root=args.cv_root,
        output_prefix=args.output_prefix,
        dpi=args.dpi,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()