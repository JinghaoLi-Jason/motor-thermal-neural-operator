"""
plot_dataset_ewma_features.py

Visualize p and EWMA load-memory features in the direct EWMA dataset.

Expected input channels:
    [u0, t_norm, x, y, Q_base, p, ewma_3, ewma_10, ewma_30, ...]

Usage:
    python plot_dataset_ewma_features.py \
        --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 \
        --output_dir results/ewma_dataset_plots \
        --conditions half 75 full

For v2:
    python plot_dataset_ewma_features.py \
        --dataset data/processed/fno_6cond_element_512_direct_ewma_v2.h5 \
        --output_dir results/ewma_dataset_plots_v2 \
        --conditions half 75 full
"""

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/ewma_dataset_plots")
    parser.add_argument("--conditions", nargs="+", default=["25", "half", "75", "full"])
    parser.add_argument("--t_end", type=float, default=60000.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


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
    return mapping.get(cond, cond)


def set_plot_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
    })


def extract_condition_features(f, cond, t_end):
    key = f"idx_{cond}"
    if key not in f:
        raise KeyError(f"{key} not found. Available: {[k for k in f.keys() if k.startswith('idx_')]}")

    idxs = f[key][:]
    X = f["X_all"]

    mask = f["mask"][:].astype(bool)

    # Input layout:
    # 0 u0, 1 t_norm, 2 x, 3 y, 4 Q_base, 5 p, 6... ewma
    in_channels = X.shape[1]
    n_ewma = in_channels - 6

    try:
        spans = json.loads(f.attrs["ewma_spans_json"])
    except Exception:
        spans = list(range(n_ewma))

    rows = []

    for sample_i, idx in enumerate(idxs):
        x_sample = X[idx]

        # These channels are spatially constant, so average inside mask
        t_norm = float(np.mean(x_sample[1][mask]))
        p_val = float(np.mean(x_sample[5][mask]))

        row = {
            "sample": sample_i,
            "idx": int(idx),
            "time_s": t_norm * t_end,
            "time_ks": t_norm * t_end / 1000.0,
            "p": p_val,
        }

        for j in range(n_ewma):
            span = spans[j] if j < len(spans) else j
            row[f"ewma_{span}"] = float(np.mean(x_sample[6 + j][mask]))

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by time just in case
    df = df.sort_values("time_s").reset_index(drop=True)

    return df, spans


def plot_condition(df, spans, cond, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(7.16, 3.4))

    ax.plot(df["time_ks"], df["p"], "k-", linewidth=2.0, label="Current load $p(t)$")

    for span in spans:
        col = f"ewma_{span}"
        if col in df.columns:
            ax.plot(df["time_ks"], df[col], label=f"EWMA {span}")

    ax.set_xlabel("Time (ks)")
    ax.set_ylabel("Load / memory value")
    ax.set_title(f"EWMA load-memory features for {condition_label(cond)} constant load")
    ax.legend(loc="best")

    fig.tight_layout()

    png_path = os.path.join(output_dir, f"ewma_features_{cond}.png")
    pdf_path = os.path.join(output_dir, f"ewma_features_{cond}.pdf")

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_all_conditions(all_dfs, spans, output_dir, dpi):
    n = len(all_dfs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, 2.4 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, (cond, df) in zip(axes, all_dfs.items()):
        ax.plot(df["time_ks"], df["p"], "k-", linewidth=2.0, label="$p(t)$")

        for span in spans:
            col = f"ewma_{span}"
            if col in df.columns:
                ax.plot(df["time_ks"], df[col], label=f"EWMA {span}")

        ax.set_title(condition_label(cond))
        ax.set_xlabel("Time (ks)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    for j in range(len(all_dfs), len(axes)):
        axes[j].axis("off")

    # One shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=True)

    fig.suptitle("EWMA load-memory features in training data", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    png_path = os.path.join(output_dir, "ewma_features_all_conditions.png")
    pdf_path = os.path.join(output_dir, "ewma_features_all_conditions.pdf")

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_plot_style()

    print("=" * 80)
    print("PLOT EWMA FEATURES FROM DATASET")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.output_dir}")
    print(f"Conditions: {args.conditions}")
    print()

    all_dfs = {}
    spans_ref = None

    with h5py.File(args.dataset, "r") as f:
        print(f"X_all shape: {f['X_all'].shape}")
        print(f"Input type: {f.attrs.get('input_type', 'unknown')}")
        print(f"EWMA spans: {f.attrs.get('ewma_spans_json', 'unknown')}")
        print()

        for cond in args.conditions:
            df, spans = extract_condition_features(f, cond, args.t_end)

            if spans_ref is None:
                spans_ref = spans

            all_dfs[cond] = df

            csv_path = os.path.join(args.output_dir, f"ewma_features_{cond}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

            plot_condition(df, spans, cond, args.output_dir, args.dpi)

    plot_all_conditions(all_dfs, spans_ref, args.output_dir, args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()