"""
visualize_direct_ewma_dataset.py

Visualize and inspect the direct-EWMA FNO HDF5 dataset.

Expected model input:
    [u0, t_norm, x, y, Q_input, p,
     EWMA_3, EWMA_10, EWMA_30]

Supported target storage:
    1. global_normalized
       Y_all contains globally standardized dT.

    2. physical
       Y_all contains physical dT in degrees Celsius.

Generated outputs:
    - dataset_summary.json
    - condition_summary.csv
    - sample_summary.csv
    - spatial_reference_fields.pdf/png
    - condition_feature_curves.pdf/png
    - condition_temperature_curves.pdf/png
    - representative_samples_<condition>.pdf/png
    - target_distribution.pdf/png
    - target_distribution_summary.csv
    - normalization_summary.pdf/png

Example:
    python visualize_dataset.py \
        --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 \
        --output_dir results/dataset_visualization \
        --dpi 600
"""

import argparse
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Condition labels and ordering
# ============================================================
CONDITION_LABELS = {
    "25": "25%",
    "half": "50%",
    "50": "50%",
    "62.5": "62.5%",
    "75": "75%",
    "87.5": "87.5%",
    "full": "100%",
    "100": "100%",
}

CONDITION_ORDER = {
    "25": 0,
    "half": 1,
    "50": 1,
    "62.5": 2,
    "75": 3,
    "87.5": 4,
    "full": 5,
    "100": 5,
}


# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a direct-EWMA FNO HDF5 dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the direct-EWMA HDF5 dataset.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/dataset_visualization",
        help="Directory for generated figures and summaries.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution.",
    )

    parser.add_argument(
        "--n_representative_times",
        type=int,
        default=4,
        help="Number of representative times shown per condition.",
    )

    parser.add_argument(
        "--histogram_samples",
        type=int,
        default=1_000_000,
        help="Maximum total active target values used in histograms.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================
# Plot style
# ============================================================
def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Liberation Serif",
            "DejaVu Serif",
        ],
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


def save_figure(fig, output_dir, name, dpi):
    png_path = os.path.join(
        output_dir,
        f"{name}.png",
    )

    pdf_path = os.path.join(
        output_dir,
        f"{name}.pdf",
    )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# JSON conversion
# ============================================================
def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    return value


# ============================================================
# Dataset helpers
# ============================================================
def get_conditions(h5_file):
    conditions = []

    for key in h5_file.keys():
        if key.startswith("idx_"):
            condition = key[4:]

            indices = h5_file[key][
                :
            ].astype(np.int64)

            conditions.append(
                (condition, indices)
            )

    conditions.sort(
        key=lambda item: (
            CONDITION_ORDER.get(
                item[0],
                999,
            ),
            item[0],
        )
    )

    if not conditions:
        raise KeyError(
            "No idx_* condition arrays were found."
        )

    return conditions


def get_channel_names(h5_file):
    n_channels = int(
        h5_file["X_all"].shape[1]
    )

    spans = []

    if "ewma_spans_json" in h5_file.attrs:
        spans = json.loads(
            h5_file.attrs[
                "ewma_spans_json"
            ]
        )

    names = [
        "u0",
        "t_norm",
        "x",
        "y",
        "Q_input",
        "p",
    ]

    names.extend(
        f"ewma_{span}"
        for span in spans
    )

    while len(names) < n_channels:
        names.append(
            f"channel_{len(names)}"
        )

    return names[:n_channels]


def determine_target_storage(h5_file):
    if "target_storage" in h5_file.attrs:
        value = h5_file.attrs[
            "target_storage"
        ]

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        return str(value)

    if "target_is_normalized" in h5_file.attrs:
        normalized = bool(
            h5_file.attrs[
                "target_is_normalized"
            ]
        )

        return (
            "global_normalized"
            if normalized
            else "physical"
        )

    # Legacy direct-EWMA dataset behavior.
    if (
        "dT_mean" in h5_file.attrs
        and "dT_std" in h5_file.attrs
    ):
        return "global_normalized"

    raise KeyError(
        "Unable to determine target storage format."
    )


def denormalize_u0(u0_normalized, h5_file):
    mean = float(
        h5_file.attrs["T_abs_mean"]
    )

    std = float(
        h5_file.attrs["T_abs_std"]
    )

    return (
        u0_normalized * std
        + mean
    )


def denormalize_q(q_normalized, h5_file):
    mean = float(
        h5_file.attrs.get(
            "Q_mean",
            0.0,
        )
    )

    std = float(
        h5_file.attrs.get(
            "Q_std",
            1.0,
        )
    )

    return (
        q_normalized * std
        + mean
    )


def target_to_physical(
    target_stored,
    h5_file,
    target_storage,
):
    if target_storage == "physical":
        return target_stored

    mean = float(
        h5_file.attrs["dT_mean"]
    )

    std = float(
        h5_file.attrs["dT_std"]
    )

    return (
        target_stored * std
        + mean
    )


def get_time_seconds(
    h5_file,
    indices,
):
    if "elapsed_time_all" in h5_file:
        return h5_file[
            "elapsed_time_all"
        ][indices].astype(np.float64)

    if "time_norm_all" in h5_file:
        t_norm = h5_file[
            "time_norm_all"
        ][indices].astype(np.float64)

    else:
        X = h5_file["X_all"]
        mask_bool = (
            h5_file["mask"][:] > 0
        )

        values = []

        for index in indices:
            t_channel = X[
                int(index),
                1,
            ]

            values.append(
                float(
                    np.mean(
                        t_channel[mask_bool]
                    )
                )
            )

        t_norm = np.asarray(
            values,
            dtype=np.float64,
        )

    duration = float(
        h5_file.attrs.get(
            "constant_t_end_s",
            21780.0,
        )
    )

    return t_norm * duration


def extract_scalar_channel(
    sample,
    channel,
    mask_bool,
):
    values = sample[
        channel
    ][mask_bool]

    return float(
        np.mean(values)
    )


# ============================================================
# Dataset validation
# ============================================================
def validate_dataset(
    h5_file,
    conditions,
):
    required = [
        "X_all",
        "Y_all",
        "mask",
        "x_grid",
        "y_grid",
    ]

    for name in required:
        if name not in h5_file:
            raise KeyError(
                f"Missing required HDF5 dataset: {name}"
            )

    X = h5_file["X_all"]
    Y = h5_file["Y_all"]
    mask = h5_file["mask"][:]

    if X.ndim != 4:
        raise ValueError(
            "X_all must have shape "
            f"[N,C,H,W], got {X.shape}."
        )

    if Y.ndim != 4 or Y.shape[1] != 1:
        raise ValueError(
            "Y_all must have shape "
            f"[N,1,H,W], got {Y.shape}."
        )

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X_all and Y_all contain different numbers of samples."
        )

    if X.shape[2:] != mask.shape:
        raise ValueError(
            "X_all spatial shape does not "
            "match the geometry mask."
        )

    if Y.shape[2:] != mask.shape:
        raise ValueError(
            "Y_all spatial shape does not "
            "match the geometry mask."
        )

    combined_indices = np.concatenate([
        indices
        for _, indices in conditions
    ])

    if len(combined_indices) != X.shape[0]:
        raise ValueError(
            "Condition indices do not cover "
            "all dataset samples."
        )

    if (
        len(np.unique(combined_indices))
        != len(combined_indices)
    ):
        raise ValueError(
            "Condition index arrays overlap."
        )

    if (
        combined_indices.min() < 0
        or combined_indices.max()
        >= X.shape[0]
    ):
        raise ValueError(
            "Condition index arrays contain "
            "out-of-range indices."
        )


# ============================================================
# Dataset summary
# ============================================================
def save_dataset_summary(
    h5_file,
    conditions,
    channel_names,
    output_dir,
):
    mask_bool = (
        h5_file["mask"][:] > 0
    )

    summary = {
        "dataset": os.path.abspath(
            h5_file.filename
        ),
        "X_all_shape": list(
            h5_file["X_all"].shape
        ),
        "Y_all_shape": list(
            h5_file["Y_all"].shape
        ),
        "X_all_dtype": str(
            h5_file["X_all"].dtype
        ),
        "Y_all_dtype": str(
            h5_file["Y_all"].dtype
        ),
        "channel_names": channel_names,
        "target_storage":
            determine_target_storage(
                h5_file
            ),
        "active_pixels": int(
            mask_bool.sum()
        ),
        "coverage_percent": float(
            mask_bool.mean() * 100.0
        ),
        "conditions": {
            condition: {
                "label":
                    CONDITION_LABELS.get(
                        condition,
                        condition,
                    ),
                "n_samples":
                    int(len(indices)),
                "first_index":
                    int(indices[0]),
                "last_index":
                    int(indices[-1]),
            }
            for condition, indices
            in conditions
        },
        "attributes": {
            key: json_safe(value)
            for key, value
            in h5_file.attrs.items()
        },
    }

    path = os.path.join(
        output_dir,
        "dataset_summary.json",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(f"Saved: {path}")


# ============================================================
# Per-sample summaries
# ============================================================
def build_sample_summaries(
    h5_file,
    conditions,
    channel_names,
):
    X = h5_file["X_all"]
    Y = h5_file["Y_all"]

    mask_bool = (
        h5_file["mask"][:] > 0
    )

    target_storage = (
        determine_target_storage(
            h5_file
        )
    )

    condition_rows = []
    sample_rows = []

    for condition, indices in conditions:
        times_s = get_time_seconds(
            h5_file,
            indices,
        )

        condition_target_min = np.inf
        condition_target_max = -np.inf

        final_T_max = None
        final_T_mean = None
        load_fraction = None

        for local_index, global_index in enumerate(
            indices
        ):
            sample = X[
                int(global_index)
            ]

            target = target_to_physical(
                Y[int(global_index), 0],
                h5_file,
                target_storage,
            )

            u0 = denormalize_u0(
                sample[0],
                h5_file,
            )

            temperature = (
                u0 + target
            )

            active_target = target[
                mask_bool
            ]

            active_temperature = temperature[
                mask_bool
            ]

            condition_target_min = min(
                condition_target_min,
                float(active_target.min()),
            )

            condition_target_max = max(
                condition_target_max,
                float(active_target.max()),
            )

            row = {
                "condition": condition,
                "condition_label":
                    CONDITION_LABELS.get(
                        condition,
                        condition,
                    ),
                "local_sample_index":
                    int(local_index),
                "global_sample_index":
                    int(global_index),
                "time_s":
                    float(times_s[local_index]),
                "time_ks":
                    float(
                        times_s[local_index]
                        / 1000.0
                    ),
                "T_max_degC":
                    float(
                        active_temperature.max()
                    ),
                "T_mean_degC":
                    float(
                        active_temperature.mean()
                    ),
                "dT_max_degC":
                    float(
                        active_target.max()
                    ),
                "dT_mean_degC":
                    float(
                        active_target.mean()
                    ),
            }

            for channel_index in range(
                5,
                X.shape[1],
            ):
                channel_name = (
                    channel_names[
                        channel_index
                    ]
                )

                row[channel_name] = (
                    extract_scalar_channel(
                        sample,
                        channel_index,
                        mask_bool,
                    )
                )

            load_fraction = row.get(
                "p",
                None,
            )

            sample_rows.append(row)

            if (
                local_index
                == len(indices) - 1
            ):
                final_T_max = float(
                    active_temperature.max()
                )

                final_T_mean = float(
                    active_temperature.mean()
                )

        condition_rows.append({
            "condition": condition,
            "condition_label":
                CONDITION_LABELS.get(
                    condition,
                    condition,
                ),
            "n_samples":
                int(len(indices)),
            "first_global_index":
                int(indices[0]),
            "last_global_index":
                int(indices[-1]),
            "time_start_s":
                float(times_s[0]),
            "time_end_s":
                float(times_s[-1]),
            "load_fraction":
                load_fraction,
            "dT_active_min_degC":
                float(condition_target_min),
            "dT_active_max_degC":
                float(condition_target_max),
            "final_T_max_degC":
                final_T_max,
            "final_T_mean_degC":
                final_T_mean,
        })

    return (
        pd.DataFrame(condition_rows),
        pd.DataFrame(sample_rows),
    )


# ============================================================
# Spatial reference fields
# ============================================================
def plot_spatial_reference_fields(
    h5_file,
    conditions,
    output_dir,
    dpi,
):
    X = h5_file["X_all"]

    mask = (
        h5_file["mask"][:] > 0
    )

    x_grid = h5_file["x_grid"][:]
    y_grid = h5_file["y_grid"][:]

    extent = [
        x_grid[0],
        x_grid[-1],
        y_grid[0],
        y_grid[-1],
    ]

    first_index = int(
        conditions[0][1][0]
    )

    sample = X[first_index]

    u0 = denormalize_u0(
        sample[0],
        h5_file,
    )

    q_input = denormalize_q(
        sample[4],
        h5_file,
    )

    fields = [
        (
            np.where(
                mask,
                u0,
                np.nan,
            ),
            r"Initial temperature $u_0$ (°C)",
            "jet",
        ),
        (
            np.where(
                mask,
                q_input,
                np.nan,
            ),
            r"Heat-source input $Q_{\mathrm{input}}$",
            "inferno",
        ),
    ]

    if (
        "Q_grid_base_physical"
        in h5_file
    ):
        fields.append((
            np.where(
                mask,
                h5_file[
                    "Q_grid_base_physical"
                ][:],
                np.nan,
            ),
            r"Base heat source $Q_{\mathrm{base}}$",
            "inferno",
        ))

    fields.extend([
        (
            np.where(
                mask,
                sample[2],
                np.nan,
            ),
            r"Normalized coordinate $x$",
            "viridis",
        ),
        (
            np.where(
                mask,
                sample[3],
                np.nan,
            ),
            r"Normalized coordinate $y$",
            "viridis",
        ),
        (
            np.where(
                mask,
                h5_file["mask"][:],
                np.nan,
            ),
            "Geometry mask",
            "gray",
        ),
    ])

    n_columns = 3

    n_rows = int(
        np.ceil(
            len(fields) / n_columns
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            7.16,
            2.15 * n_rows,
        ),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for axis, (
        field,
        title,
        color_map,
    ) in zip(axes, fields):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap=color_map,
            aspect="equal",
        )

        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

        fig.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.02,
        )

    for axis in axes[
        len(fields):
    ]:
        axis.axis("off")

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "spatial_reference_fields",
        dpi,
    )


# ============================================================
# EWMA feature curves
# ============================================================
def plot_condition_feature_curves(
    sample_df,
    output_dir,
    dpi,
):
    ewma_columns = [
        column
        for column
        in sample_df.columns
        if column.startswith("ewma_")
    ]

    conditions = list(
        sample_df[
            "condition"
        ].drop_duplicates()
    )

    fig, axes = plt.subplots(
        len(conditions),
        1,
        figsize=(
            3.5,
            1.55 * len(conditions),
        ),
        sharex=False,
    )

    axes = np.atleast_1d(axes)

    for axis, condition in zip(
        axes,
        conditions,
    ):
        frame = sample_df[
            sample_df["condition"]
            == condition
        ]

        axis.plot(
            frame["time_ks"],
            frame["p"],
            "k-",
            linewidth=1.4,
            label=r"Current load $p_i$",
        )

        for column in ewma_columns:
            axis.plot(
                frame["time_ks"],
                frame[column],
                label=column.replace(
                    "_",
                    " ",
                ),
            )

        axis.set_ylabel(
            CONDITION_LABELS.get(
                condition,
                condition,
            )
        )

        axis.grid(
            True,
            alpha=0.3,
        )

        if axis is axes[-1]:
            axis.legend(
                ncol=2,
                loc="best",
            )

    axes[-1].set_xlabel(
        "Elapsed time (ks)"
    )

    fig.suptitle(
        "Load and EWMA features by condition"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    save_figure(
        fig,
        output_dir,
        "condition_feature_curves",
        dpi,
    )


# ============================================================
# Temperature evolution
# ============================================================
def plot_condition_temperature_curves(
    sample_df,
    output_dir,
    dpi,
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.5, 4.0),
        sharex=True,
    )

    conditions = sample_df[
        "condition"
    ].drop_duplicates()

    for condition in conditions:
        frame = sample_df[
            sample_df["condition"]
            == condition
        ]

        label = CONDITION_LABELS.get(
            condition,
            condition,
        )

        axes[0].plot(
            frame["time_ks"],
            frame["T_max_degC"],
            label=label,
        )

        axes[1].plot(
            frame["time_ks"],
            frame["T_mean_degC"],
            label=label,
        )

    axes[0].set_ylabel(
        r"$T_{\max}$ (°C)"
    )

    axes[1].set_ylabel(
        r"$T_{\mathrm{mean}}$ (°C)"
    )

    axes[1].set_xlabel(
        "Elapsed time (ks)"
    )

    for axis in axes:
        axis.grid(
            True,
            alpha=0.3,
        )

    axes[0].legend(
        ncol=2,
        loc="best",
    )

    fig.suptitle(
        "Temperature evolution in the processed dataset"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    save_figure(
        fig,
        output_dir,
        "condition_temperature_curves",
        dpi,
    )


# ============================================================
# Representative fields
# ============================================================
def plot_representative_samples(
    h5_file,
    conditions,
    output_dir,
    dpi,
    n_representative_times,
):
    X = h5_file["X_all"]
    Y = h5_file["Y_all"]

    mask = (
        h5_file["mask"][:] > 0
    )

    x_grid = h5_file["x_grid"][:]
    y_grid = h5_file["y_grid"][:]

    extent = [
        x_grid[0],
        x_grid[-1],
        y_grid[0],
        y_grid[-1],
    ]

    target_storage = (
        determine_target_storage(
            h5_file
        )
    )

    for condition, indices in conditions:
        n_selected = min(
            max(
                n_representative_times,
                1,
            ),
            len(indices),
        )

        local_indices = np.linspace(
            0,
            len(indices) - 1,
            n_selected,
            dtype=int,
        )

        selected_indices = indices[
            local_indices
        ]

        times_s = get_time_seconds(
            h5_file,
            selected_indices,
        )

        temperature_fields = []
        target_fields = []
        q_fields = []

        for global_index in selected_indices:
            sample = X[
                int(global_index)
            ]

            u0 = denormalize_u0(
                sample[0],
                h5_file,
            )

            target = target_to_physical(
                Y[int(global_index), 0],
                h5_file,
                target_storage,
            )

            temperature_fields.append(
                u0 + target
            )

            target_fields.append(
                target
            )

            q_fields.append(
                denormalize_q(
                    sample[4],
                    h5_file,
                )
            )

        temperature_fields = np.asarray(
            temperature_fields
        )

        target_fields = np.asarray(
            target_fields
        )

        q_fields = np.asarray(
            q_fields
        )

        temperature_min = float(
            temperature_fields[
                :, mask
            ].min()
        )

        temperature_max = float(
            temperature_fields[
                :, mask
            ].max()
        )

        target_limit = max(
            1.0,
            float(
                np.max(
                    np.abs(
                        target_fields[
                            :, mask
                        ]
                    )
                )
            ),
        )

        q_min = float(
            q_fields[:, mask].min()
        )

        q_max = float(
            q_fields[:, mask].max()
        )

        fig, axes = plt.subplots(
            3,
            n_selected,
            figsize=(7.16, 4.45),
            squeeze=False,
        )

        for column, time_s in enumerate(
            times_s
        ):
            temperature_display = (
                np.where(
                    mask,
                    temperature_fields[
                        column
                    ],
                    np.nan,
                )
            )

            target_display = np.where(
                mask,
                target_fields[column],
                np.nan,
            )

            q_display = np.where(
                mask,
                q_fields[column],
                np.nan,
            )

            image_temperature = (
                axes[0, column].imshow(
                    temperature_display,
                    origin="lower",
                    extent=extent,
                    cmap="jet",
                    vmin=temperature_min,
                    vmax=temperature_max,
                    aspect="equal",
                )
            )

            image_target = (
                axes[1, column].imshow(
                    target_display,
                    origin="lower",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-target_limit,
                    vmax=target_limit,
                    aspect="equal",
                )
            )

            image_q = (
                axes[2, column].imshow(
                    q_display,
                    origin="lower",
                    extent=extent,
                    cmap="inferno",
                    vmin=q_min,
                    vmax=q_max,
                    aspect="equal",
                )
            )

            axes[0, column].set_title(
                f"$t={time_s / 1000:.2f}$ ks"
            )

            for row in range(3):
                axes[
                    row,
                    column,
                ].set_xticks([])

                axes[
                    row,
                    column,
                ].set_yticks([])

        axes[0, 0].set_ylabel(
            r"$T$",
            fontweight="bold",
        )

        axes[1, 0].set_ylabel(
            r"$\Delta T$",
            fontweight="bold",
        )

        axes[2, 0].set_ylabel(
            r"$Q_{\mathrm{input}}$",
            fontweight="bold",
        )

        fig.suptitle(
            "Representative processed fields: "
            + CONDITION_LABELS.get(
                condition,
                condition,
            )
        )

        fig.subplots_adjust(
            left=0.055,
            right=0.90,
            top=0.90,
            bottom=0.04,
            hspace=0.22,
            wspace=0.04,
        )

        temperature_cbar_axis = (
            fig.add_axes([
                0.915,
                0.665,
                0.012,
                0.22,
            ])
        )

        target_cbar_axis = (
            fig.add_axes([
                0.915,
                0.370,
                0.012,
                0.22,
            ])
        )

        q_cbar_axis = fig.add_axes([
            0.915,
            0.075,
            0.012,
            0.22,
        ])

        fig.colorbar(
            image_temperature,
            cax=temperature_cbar_axis,
            label="Temperature (°C)",
        )

        fig.colorbar(
            image_target,
            cax=target_cbar_axis,
            label=r"$\Delta T$ (°C)",
        )

        fig.colorbar(
            image_q,
            cax=q_cbar_axis,
            label=r"$Q_{\mathrm{input}}$",
        )

        safe_condition = (
            condition.replace(
                ".",
                "_",
            )
        )

        save_figure(
            fig,
            output_dir,
            (
                "representative_samples_"
                f"{safe_condition}"
            ),
            dpi,
        )


# ============================================================
# Target distribution
# ============================================================
def plot_target_distribution(
    h5_file,
    conditions,
    output_dir,
    dpi,
    maximum_values,
    seed,
):
    Y = h5_file["Y_all"]

    mask_bool = (
        h5_file["mask"][:] > 0
    )

    target_storage = (
        determine_target_storage(
            h5_file
        )
    )

    random_generator = (
        np.random.default_rng(seed)
    )

    values_per_condition = max(
        int(
            maximum_values
            / max(len(conditions), 1)
        ),
        1000,
    )

    statistics = []

    fig, axis = plt.subplots(
        figsize=(3.5, 2.7)
    )

    for condition, indices in conditions:
        collected = []
        remaining = values_per_condition

        for global_index in indices:
            target = target_to_physical(
                Y[int(global_index), 0],
                h5_file,
                target_storage,
            )

            active_values = target[
                mask_bool
            ]

            if active_values.size > remaining:
                selected = (
                    random_generator.choice(
                        active_values,
                        size=remaining,
                        replace=False,
                    )
                )

            else:
                selected = active_values

            collected.append(selected)
            remaining -= len(selected)

            if remaining <= 0:
                break

        values = np.concatenate(
            collected
        )

        statistics.append({
            "condition": condition,
            "condition_label":
                CONDITION_LABELS.get(
                    condition,
                    condition,
                ),
            "n_values": int(len(values)),
            "mean_degC":
                float(values.mean()),
            "std_degC":
                float(values.std()),
            "min_degC":
                float(values.min()),
            "max_degC":
                float(values.max()),
        })

        axis.hist(
            values,
            bins=80,
            density=True,
            histtype="step",
            label=CONDITION_LABELS.get(
                condition,
                condition,
            ),
        )

    axis.set_xlabel(
        r"Temperature rise $\Delta T$ (°C)"
    )

    axis.set_ylabel(
        "Probability density"
    )

    axis.set_title(
        "Active-domain target distribution"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend(
        ncol=2,
        loc="best",
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "target_distribution",
        dpi,
    )

    path = os.path.join(
        output_dir,
        "target_distribution_summary.csv",
    )

    pd.DataFrame(
        statistics
    ).to_csv(
        path,
        index=False,
    )

    print(f"Saved: {path}")


# ============================================================
# Normalization summary
# ============================================================
def plot_normalization_summary(
    h5_file,
    output_dir,
    dpi,
):
    labels = []
    means = []
    standard_deviations = []

    attribute_pairs = [
        (
            r"$u_0$",
            "T_abs_mean",
            "T_abs_std",
        ),
        (
            r"$Q$",
            "Q_mean",
            "Q_std",
        ),
        (
            r"$\Delta T$",
            "dT_mean",
            "dT_std",
        ),
    ]

    for (
        label,
        mean_attribute,
        std_attribute,
    ) in attribute_pairs:
        if (
            mean_attribute
            in h5_file.attrs
            and std_attribute
            in h5_file.attrs
        ):
            labels.append(label)

            means.append(
                float(
                    h5_file.attrs[
                        mean_attribute
                    ]
                )
            )

            standard_deviations.append(
                float(
                    h5_file.attrs[
                        std_attribute
                    ]
                )
            )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.6),
    )

    positions = np.arange(
        len(labels)
    )

    axes[0].bar(
        positions,
        means,
    )

    axes[0].set_xticks(
        positions
    )

    axes[0].set_xticklabels(
        labels
    )

    axes[0].set_ylabel(
        "Mean"
    )

    axes[0].set_title(
        "Stored normalization means"
    )

    axes[0].grid(
        axis="y",
        alpha=0.3,
    )

    axes[1].bar(
        positions,
        standard_deviations,
    )

    axes[1].set_xticks(
        positions
    )

    axes[1].set_xticklabels(
        labels
    )

    axes[1].set_ylabel(
        "Standard deviation"
    )

    axes[1].set_title(
        "Stored normalization standard deviations"
    )

    axes[1].grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "normalization_summary",
        dpi,
    )


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    set_plot_style()

    print("=" * 80)
    print("DIRECT EWMA DATASET VISUALIZATION")
    print("=" * 80)
    print(f"Dataset:    {args.dataset}")
    print(f"Output dir: {args.output_dir}")
    print()

    with h5py.File(
        args.dataset,
        "r",
    ) as h5_file:
        conditions = get_conditions(
            h5_file
        )

        validate_dataset(
            h5_file,
            conditions,
        )

        channel_names = get_channel_names(
            h5_file
        )

        target_storage = (
            determine_target_storage(
                h5_file
            )
        )

        print(
            f"X_all shape: "
            f"{h5_file['X_all'].shape}"
        )

        print(
            f"Y_all shape: "
            f"{h5_file['Y_all'].shape}"
        )

        print(
            f"Input channels: "
            f"{channel_names}"
        )

        print(
            f"Target storage: "
            f"{target_storage}"
        )

        print("Conditions:")

        for condition, indices in conditions:
            print(
                f"  "
                f"{CONDITION_LABELS.get(condition, condition):>6}: "
                f"{len(indices)} samples, "
                f"indices {indices[0]}–{indices[-1]}"
            )

        print()

        (
            condition_summary,
            sample_summary,
        ) = build_sample_summaries(
            h5_file,
            conditions,
            channel_names,
        )

        condition_summary_path = os.path.join(
            args.output_dir,
            "condition_summary.csv",
        )

        condition_summary.to_csv(
            condition_summary_path,
            index=False,
        )

        print(
            f"Saved: "
            f"{condition_summary_path}"
        )

        sample_summary_path = os.path.join(
            args.output_dir,
            "sample_summary.csv",
        )

        sample_summary.to_csv(
            sample_summary_path,
            index=False,
        )

        print(
            f"Saved: "
            f"{sample_summary_path}"
        )

        save_dataset_summary(
            h5_file,
            conditions,
            channel_names,
            args.output_dir,
        )

        plot_spatial_reference_fields(
            h5_file,
            conditions,
            args.output_dir,
            args.dpi,
        )

        plot_condition_feature_curves(
            sample_summary,
            args.output_dir,
            args.dpi,
        )

        plot_condition_temperature_curves(
            sample_summary,
            args.output_dir,
            args.dpi,
        )

        plot_representative_samples(
            h5_file,
            conditions,
            args.output_dir,
            args.dpi,
            args.n_representative_times,
        )

        plot_target_distribution(
            h5_file,
            conditions,
            args.output_dir,
            args.dpi,
            args.histogram_samples,
            args.seed,
        )

        plot_normalization_summary(
            h5_file,
            args.output_dir,
            args.dpi,
        )

    print()
    print("Done.")
    print(
        "Saved all dataset visualizations to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()