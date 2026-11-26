"""
train_direct_ewma.py

Train a direct FNO with EWMA load-memory inputs.

Input:
    [u0, t_norm, x, y, Q_base, p, ewma_3, ewma_10, ewma_30]

Output:
    dT(t) = T(t) - u0

Scenarios:
    final_all
    loco_trainall
    train_val
    loco_val

Important protocol details:
    - Target normalization statistics are calculated from training
      samples only.
    - Inactive pixels outside the motor geometry are excluded from
      normalization-statistic calculation.
    - Validation controls checkpoint selection and early stopping.
    - The held-out test condition is evaluated only once, after model
      selection.
    - Test performance is never used for checkpoint selection,
      early stopping, scheduling, or hyperparameter selection.

Example final-model run:
    python train_direct_ewma.py \
        --scenario final_all \
        --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 \
        --output_dir results/direct_ewma_model/final_50_75_100 \
        --include_conditions half,75,full \
        --device cuda:1

Example LOCO-validation run:
    python train_direct_ewma.py \
        --scenario loco_val \
        --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 \
        --output_dir results/direct_ewma_cv/heldout_25 \
        --include_conditions 25,half,75,full \
        --test_condition 25 \
        --device cuda:1
"""

import argparse
import json
import os
import random
import time

import h5py
import numpy as np
import torch
from neuralop.models import FNO


# ============================================================
# Command-line arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/fno_6cond_element_512_direct_ewma.h5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/direct_ewma_model",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=[
            "final_all",
            "loco_trainall",
            "train_val",
            "loco_val",
        ],
    )
    parser.add_argument(
        "--include_conditions",
        type=str,
        default="half,75,full",
    )
    parser.add_argument(
        "--test_condition",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--val_tail",
        type=int,
        default=19,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--modes",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "none"],
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Condition-name helpers
# ============================================================
def parse_condition_list(value):
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def normalize_condition_name(name):
    mapping = {
        "25": "25",
        "50": "half",
        "half": "half",
        "75": "75",
        "100": "full",
        "full": "full",
        "62.5": "62.5",
        "87.5": "87.5",
    }

    normalized_name = str(name).strip()
    return mapping.get(normalized_name, normalized_name)


# ============================================================
# Device information
# ============================================================
def get_device_info(device):
    info = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "pytorch_version": str(torch.__version__),
        "pytorch_cuda_version": (
            str(torch.version.cuda)
            if torch.version.cuda is not None
            else None
        ),
    }

    if device.type == "cuda":
        gpu_index = device.index

        if gpu_index is None:
            gpu_index = torch.cuda.current_device()

        properties = torch.cuda.get_device_properties(gpu_index)

        info.update({
            "gpu_index": int(gpu_index),
            "gpu_name": torch.cuda.get_device_name(gpu_index),
            "gpu_total_memory_bytes": int(properties.total_memory),
            "gpu_total_memory_gib": (
                float(properties.total_memory) / 1024.0**3
            ),
        })
    else:
        info.update({
            "gpu_index": None,
            "gpu_name": None,
            "gpu_total_memory_bytes": None,
            "gpu_total_memory_gib": None,
        })

    return info


def print_device_info(device):
    info = get_device_info(device)

    print(f"Selected device: {info['device']}")

    if device.type == "cuda":
        print(f"GPU index: {info['gpu_index']}")
        print(f"GPU name: {info['gpu_name']}")
        print(
            "GPU memory: "
            f"{info['gpu_total_memory_gib']:.2f} GiB"
        )
        print(
            "PyTorch CUDA version: "
            f"{info['pytorch_cuda_version']}"
        )
    else:
        print("CUDA is unavailable or CPU was requested.")


# ============================================================
# Masked loss functions
# ============================================================
def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask.unsqueeze(0).unsqueeze(0)

    denominator = mask.sum() * pred.shape[0]
    return diff2.sum() / denominator


def masked_mae(pred, target, mask):
    diff = torch.abs(pred - target)
    diff = diff * mask.unsqueeze(0).unsqueeze(0)

    denominator = mask.sum() * pred.shape[0]
    return diff.sum() / denominator


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(
    model,
    X_eval,
    Y_eval,
    mask,
    dT_mean,
    dT_std,
    batch_size,
):
    model.eval()

    total_sq = 0.0
    total_abs = 0.0
    total_pixels = 0.0
    total_loss = 0.0
    n_samples = 0

    for start in range(0, len(X_eval), batch_size):
        X = X_eval[start:start + batch_size]
        Y = Y_eval[start:start + batch_size]

        batch_size_actual = X.shape[0]

        pred = model(X)
        loss = masked_mse(pred, Y, mask)

        # Convert fold-normalized values back to physical dT in degrees C.
        pred_phys = pred * dT_std + dT_mean
        Y_phys = Y * dT_std + dT_mean

        diff = pred_phys - Y_phys
        diff = diff * mask.unsqueeze(0).unsqueeze(0)

        total_sq += (diff ** 2).sum().item()
        total_abs += torch.abs(diff).sum().item()
        total_pixels += (
            batch_size_actual * mask.sum().item()
        )

        total_loss += loss.item() * batch_size_actual
        n_samples += batch_size_actual

    return {
        "loss": total_loss / max(n_samples, 1),
        "rmse_degC": float(
            (total_sq / max(total_pixels, 1.0)) ** 0.5
        ),
        "mae_degC": float(
            total_abs / max(total_pixels, 1.0)
        ),
    }


# ============================================================
# Dataset splitting
# ============================================================
def build_splits(
    h5_path,
    scenario,
    include_conditions,
    test_condition=None,
    val_tail=19,
):
    include_conditions = [
        normalize_condition_name(condition)
        for condition in include_conditions
    ]

    if test_condition is not None:
        test_condition = normalize_condition_name(test_condition)

    if scenario in ["loco_trainall", "loco_val"]:
        if test_condition is None:
            raise ValueError(
                f"--test_condition is required for {scenario}."
            )

        if test_condition not in include_conditions:
            raise ValueError(
                f"Test condition '{test_condition}' is not present in "
                f"--include_conditions={include_conditions}."
            )

    train_idx = []
    val_idx = []
    test_idx = []
    summary = {}

    with h5py.File(h5_path, "r") as h5_file:
        available = sorted([
            key[4:]
            for key in h5_file.keys()
            if key.startswith("idx_")
        ])

        print(f"Available conditions: {available}")

        for condition in include_conditions:
            key = f"idx_{condition}"

            if key not in h5_file:
                raise KeyError(
                    f"Condition index dataset '{key}' is missing."
                )

            indices = h5_file[key][:].astype(np.int64)
            n_samples = len(indices)

            if scenario == "final_all":
                train_idx.extend(indices.tolist())

                summary[condition] = {
                    "role": "train",
                    "train": n_samples,
                    "val": 0,
                    "test": 0,
                }

            elif scenario == "loco_trainall":
                if condition == test_condition:
                    test_idx.extend(indices.tolist())

                    summary[condition] = {
                        "role": "test",
                        "train": 0,
                        "val": 0,
                        "test": n_samples,
                    }
                else:
                    train_idx.extend(indices.tolist())

                    summary[condition] = {
                        "role": "train",
                        "train": n_samples,
                        "val": 0,
                        "test": 0,
                    }

            elif scenario == "train_val":
                if n_samples <= val_tail:
                    raise ValueError(
                        f"Condition '{condition}' has {n_samples} samples, "
                        f"but val_tail={val_tail}. At least one training "
                        "sample must remain."
                    )

                train_indices = indices[:-val_tail]
                validation_indices = indices[-val_tail:]

                train_idx.extend(train_indices.tolist())
                val_idx.extend(validation_indices.tolist())

                summary[condition] = {
                    "role": "train_val",
                    "train": len(train_indices),
                    "val": len(validation_indices),
                    "test": 0,
                }

            elif scenario == "loco_val":
                if condition == test_condition:
                    test_idx.extend(indices.tolist())

                    summary[condition] = {
                        "role": "test",
                        "train": 0,
                        "val": 0,
                        "test": n_samples,
                    }
                else:
                    if n_samples <= val_tail:
                        raise ValueError(
                            f"Condition '{condition}' has {n_samples} "
                            f"samples, but val_tail={val_tail}. At least "
                            "one training sample must remain."
                        )

                    train_indices = indices[:-val_tail]
                    validation_indices = indices[-val_tail:]

                    train_idx.extend(train_indices.tolist())
                    val_idx.extend(validation_indices.tolist())

                    summary[condition] = {
                        "role": "train_val",
                        "train": len(train_indices),
                        "val": len(validation_indices),
                        "test": 0,
                    }

    train_conditions = [
        condition
        for condition, condition_summary in summary.items()
        if condition_summary["train"] > 0
    ]

    val_conditions = [
        condition
        for condition, condition_summary in summary.items()
        if condition_summary["val"] > 0
    ]

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
        summary,
        train_conditions,
        val_conditions,
        test_condition,
    )


# ============================================================
# Fold-specific target normalization
# ============================================================
def compute_training_target_statistics(
    Y_all_global_norm,
    train_idx,
    mask_cpu,
    global_dT_mean,
    global_dT_std,
    chunk_size=8,
):
    """
    Calculate physical dT mean and standard deviation from training
    samples and active motor pixels only.

    The HDF5 target is assumed to have been standardized as:

        Y_global_norm = (dT_phys - global_mean) / global_std

    We calculate the training-only statistics in the existing
    globally normalized space and transform those statistics into
    physical units.

    This avoids constructing another complete physical-unit target
    tensor in memory.
    """
    if len(train_idx) == 0:
        raise ValueError(
            "Cannot calculate normalization statistics because "
            "the training split is empty."
        )

    if Y_all_global_norm.ndim != 4:
        raise ValueError(
            "Expected Y_all to have shape [N, C, H, W], but received "
            f"{tuple(Y_all_global_norm.shape)}."
        )

    if Y_all_global_norm.shape[1] != 1:
        raise ValueError(
            "Expected one target channel, but received "
            f"{Y_all_global_norm.shape[1]}."
        )

    mask_bool = mask_cpu > 0

    if int(mask_bool.sum().item()) == 0:
        raise ValueError("The active-domain mask contains no active pixels.")

    running_sum = 0.0
    running_sum_sq = 0.0
    running_count = 0

    train_idx_tensor = torch.as_tensor(
        train_idx,
        dtype=torch.long,
    )

    print(
        "Calculating fold-specific target normalization from "
        "training samples only..."
    )

    for start in range(0, len(train_idx_tensor), chunk_size):
        batch_indices = train_idx_tensor[start:start + chunk_size]

        # Shape: [B, 1, H, W]
        batch = Y_all_global_norm[batch_indices]

        # Shape after indexing: [B, active_pixels]
        active_values = batch[:, 0, mask_bool]

        active_values_64 = active_values.to(torch.float64)

        running_sum += active_values_64.sum().item()
        running_sum_sq += (
            active_values_64 * active_values_64
        ).sum().item()
        running_count += active_values_64.numel()

    if running_count == 0:
        raise RuntimeError(
            "No active training pixels were available for "
            "normalization."
        )

    normalized_mean = running_sum / running_count

    normalized_variance = (
        running_sum_sq / running_count
        - normalized_mean**2
    )
    normalized_variance = max(normalized_variance, 0.0)
    normalized_std = normalized_variance**0.5

    # Transform statistics from the old normalized space back to
    # physical degrees Celsius.
    fold_dT_mean = (
        global_dT_mean
        + global_dT_std * normalized_mean
    )
    fold_dT_std = global_dT_std * normalized_std

    # Numerical guard for a degenerate target distribution.
    fold_dT_std = max(float(fold_dT_std), 1e-8)

    statistics = {
        "source": "training_samples_and_active_pixels_only",
        "method": "population_mean_and_standard_deviation",
        "active_pixel_count_across_training_samples": int(
            running_count
        ),
        "training_sample_count": int(len(train_idx)),
        "dataset_global_dT_mean": float(global_dT_mean),
        "dataset_global_dT_std": float(global_dT_std),
        "training_mean_in_dataset_normalized_space": float(
            normalized_mean
        ),
        "training_std_in_dataset_normalized_space": float(
            normalized_std
        ),
        "fold_dT_mean": float(fold_dT_mean),
        "fold_dT_std": float(fold_dT_std),
    }

    return float(fold_dT_mean), float(fold_dT_std), statistics


def convert_global_to_fold_normalization(
    Y_global_norm,
    global_dT_mean,
    global_dT_std,
    fold_dT_mean,
    fold_dT_std,
):
    """
    Convert targets from dataset-global normalization to fold-specific
    normalization without first storing a full physical-unit tensor.

    Starting from:
        Y_global = (Y_phys - global_mean) / global_std

    The desired target is:
        Y_fold = (Y_phys - fold_mean) / fold_std

    Therefore:
        Y_fold =
            (Y_global * global_std + global_mean - fold_mean)
            / fold_std
    """
    return (
        Y_global_norm * global_dT_std
        + global_dT_mean
        - fold_dT_mean
    ) / fold_dT_std


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "WARNING: CUDA was requested but is unavailable. "
            "Falling back to CPU."
        )
        args.device = "cpu"

    device = torch.device(args.device)

    include_conditions = [
        normalize_condition_name(condition)
        for condition in parse_condition_list(
            args.include_conditions
        )
    ]

    print("=" * 80)
    print("TRAIN DIRECT EWMA FNO")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Scenario: {args.scenario}")
    print(f"Conditions: {include_conditions}")
    print(f"Output: {args.output_dir}")
    print()

    print_device_info(device)
    print()

    # --------------------------------------------------------
    # Read dataset metadata
    # --------------------------------------------------------
    with h5py.File(args.dataset, "r") as h5_file:
        mask_np = h5_file["mask"][:].astype(np.float32)

        resolution = int(h5_file.attrs["resolution"])
        in_channels = int(h5_file["X_all"].shape[1])

        # These are the original global dataset statistics.
        dataset_global_dT_mean = float(
            h5_file.attrs["dT_mean"]
        )
        dataset_global_dT_std = float(
            h5_file.attrs["dT_std"]
        )

        T_abs_mean = float(h5_file.attrs["T_abs_mean"])
        T_abs_std = float(h5_file.attrs["T_abs_std"])

        ewma_spans = json.loads(
            h5_file.attrs["ewma_spans_json"]
        )

    mask_cpu = torch.tensor(
        mask_np,
        dtype=torch.float32,
    )
    mask = mask_cpu.to(device)

    print(f"Resolution: {resolution}")
    print(f"Input channels: {in_channels}")
    print(
        "Original dataset-global dT mean/std: "
        f"{dataset_global_dT_mean:.6f}/"
        f"{dataset_global_dT_std:.6f}"
    )
    print(f"EWMA spans: {ewma_spans}")
    print()

    # --------------------------------------------------------
    # Construct train/validation/test splits
    # --------------------------------------------------------
    (
        train_idx,
        val_idx,
        test_idx,
        split_summary,
        train_conditions,
        val_conditions,
        test_condition,
    ) = build_splits(
        h5_path=args.dataset,
        scenario=args.scenario,
        include_conditions=include_conditions,
        test_condition=args.test_condition,
        val_tail=args.val_tail,
    )

    if len(train_idx) == 0:
        raise ValueError("The training split is empty.")

    print("--- Split ---")

    for condition, condition_summary in split_summary.items():
        print(f"{condition:>6}: {condition_summary}")

    print(
        f"TOTAL train={len(train_idx)}, "
        f"val={len(val_idx)}, "
        f"test={len(test_idx)}"
    )
    print()

    # --------------------------------------------------------
    # Load complete arrays on CPU
    # --------------------------------------------------------
    print("Loading dataset arrays...")

    with h5py.File(args.dataset, "r") as h5_file:
        X_all = torch.tensor(
            h5_file["X_all"][:],
            dtype=torch.float32,
        )
        Y_all_global_norm = torch.tensor(
            h5_file["Y_all"][:],
            dtype=torch.float32,
        )

    print(f"X_all shape: {tuple(X_all.shape)}")
    print(f"Y_all shape: {tuple(Y_all_global_norm.shape)}")
    print()

    # --------------------------------------------------------
    # Calculate training-only target normalization
    # --------------------------------------------------------
    (
        fold_dT_mean,
        fold_dT_std,
        normalization_info,
    ) = compute_training_target_statistics(
        Y_all_global_norm=Y_all_global_norm,
        train_idx=train_idx,
        mask_cpu=mask_cpu,
        global_dT_mean=dataset_global_dT_mean,
        global_dT_std=dataset_global_dT_std,
        chunk_size=args.batch_size,
    )

    print(
        "Fold-specific training dT mean/std: "
        f"{fold_dT_mean:.6f}/{fold_dT_std:.6f}"
    )
    print(
        "Normalization source: training samples and "
        "active motor pixels only"
    )
    print()

    # --------------------------------------------------------
    # Build split tensors using fold-specific normalization
    # --------------------------------------------------------
    print("Preparing training tensors...")

    X_train = X_all[train_idx].to(device)

    Y_train = convert_global_to_fold_normalization(
        Y_global_norm=Y_all_global_norm[train_idx],
        global_dT_mean=dataset_global_dT_mean,
        global_dT_std=dataset_global_dT_std,
        fold_dT_mean=fold_dT_mean,
        fold_dT_std=fold_dT_std,
    ).to(device)

    if len(val_idx) > 0:
        X_val = X_all[val_idx].to(device)

        Y_val = convert_global_to_fold_normalization(
            Y_global_norm=Y_all_global_norm[val_idx],
            global_dT_mean=dataset_global_dT_mean,
            global_dT_std=dataset_global_dT_std,
            fold_dT_mean=fold_dT_mean,
            fold_dT_std=fold_dT_std,
        ).to(device)
    else:
        X_val = None
        Y_val = None

    if len(test_idx) > 0:
        X_test = X_all[test_idx].to(device)

        Y_test = convert_global_to_fold_normalization(
            Y_global_norm=Y_all_global_norm[test_idx],
            global_dT_mean=dataset_global_dT_mean,
            global_dT_std=dataset_global_dT_std,
            fold_dT_mean=fold_dT_mean,
            fold_dT_std=fold_dT_std,
        ).to(device)
    else:
        X_test = None
        Y_test = None

    del X_all
    del Y_all_global_norm

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = FNO(
        n_modes=(args.modes, args.modes),
        in_channels=in_channels,
        out_channels=1,
        hidden_channels=args.width,
        n_layers=args.n_layers,
    ).to(device)

    n_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {n_params:,}")
    print(f"Model parameter device: {next(model.parameters()).device}")
    print()

    # --------------------------------------------------------
    # Optimizer and scheduler
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01,
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
    else:
        scheduler = None

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------
    config = {
        "dataset": args.dataset,
        "scenario": args.scenario,
        "include_conditions": include_conditions,
        "train_conditions": train_conditions,
        "val_conditions": val_conditions,
        "test_condition": (
            normalize_condition_name(args.test_condition)
            if args.test_condition is not None
            else None
        ),
        "split_summary": split_summary,
        "val_tail": args.val_tail,
        "seed": args.seed,
        "modes": args.modes,
        "width": args.width,
        "n_layers": args.n_layers,
        "input_channels": in_channels,
        "output_channels": 1,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "grad_clip": args.grad_clip,
        "early_stop_patience": args.early_stop_patience,
        "n_params": int(n_params),
        "input_type": "direct_ewma",
        "target": "dT",
        # These are the statistics that must be used to denormalize
        # this model's outputs.
        "dT_mean": fold_dT_mean,
        "dT_std": fold_dT_std,
        "normalization": normalization_info,
        "T_abs_mean": T_abs_mean,
        "T_abs_std": T_abs_std,
        "ewma_spans": ewma_spans,
        "device_info": get_device_info(device),
        "test_evaluated_during_training": False,
    }

    config_path = os.path.join(
        args.output_dir,
        "config.json",
    )

    with open(config_path, "w") as file:
        json.dump(config, file, indent=2)

    normalization_path = os.path.join(
        args.output_dir,
        "normalization.json",
    )

    with open(normalization_path, "w") as file:
        json.dump(normalization_info, file, indent=2)

    np.save(
        os.path.join(args.output_dir, "train_indices.npy"),
        train_idx,
    )

    if len(val_idx) > 0:
        np.save(
            os.path.join(args.output_dir, "val_indices.npy"),
            val_idx,
        )

    if len(test_idx) > 0:
        np.save(
            os.path.join(args.output_dir, "test_indices.npy"),
            test_idx,
        )

    print(f"Saved configuration: {config_path}")
    print(f"Saved normalization: {normalization_path}")
    print()

    # ========================================================
    # Training
    # ========================================================
    print("--- Training ---")

    history = []
    n_train = len(X_train)

    best_val = float("inf")
    best_epoch = -1
    patience = 0
    epochs_completed = 0

    # Synchronize before starting because CUDA work is asynchronous.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epochs_completed = epoch

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------
        model.train()

        permutation = torch.randperm(
            n_train,
            device=device,
        )

        loss_sum = 0.0
        mae_sum = 0.0
        n_batches = 0

        for start in range(
            0,
            n_train,
            args.batch_size,
        ):
            batch_indices = permutation[
                start:start + args.batch_size
            ]

            X = X_train[batch_indices]
            Y = Y_train[batch_indices]

            optimizer.zero_grad(set_to_none=True)

            pred = model(X)

            loss = masked_mse(pred, Y, mask)
            mae = masked_mae(pred, Y, mask)

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            loss_sum += loss.item()
            mae_sum += mae.item()
            n_batches += 1

        train_loss = loss_sum / max(n_batches, 1)
        train_mae = mae_sum / max(n_batches, 1)

        # ----------------------------------------------------
        # Validation monitoring
        #
        # Validation may control:
        #   - checkpoint selection
        #   - early stopping
        #   - plateau scheduling
        #
        # The held-out test condition is not evaluated here.
        # ----------------------------------------------------
        if X_val is not None:
            val_metrics = evaluate(
                model=model,
                X_eval=X_val,
                Y_eval=Y_val,
                mask=mask,
                dT_mean=fold_dT_mean,
                dT_std=fold_dT_std,
                batch_size=args.batch_size,
            )
        else:
            val_metrics = None

        # ----------------------------------------------------
        # Learning-rate scheduler
        # ----------------------------------------------------
        if scheduler is not None:
            if args.scheduler == "plateau":
                if val_metrics is not None:
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step(train_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Save epoch history
        # ----------------------------------------------------
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae_norm": train_mae,
            "lr": current_lr,
        }

        if val_metrics is not None:
            history_row.update({
                "val_loss": val_metrics["loss"],
                "val_rmse_degC": val_metrics["rmse_degC"],
                "val_mae_degC": val_metrics["mae_degC"],
            })

        history.append(history_row)

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------
        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == args.epochs
        ):
            message = (
                f"[{epoch:03d}/{args.epochs}] "
                f"train_loss={train_loss:.6f}"
            )

            if val_metrics is not None:
                message += (
                    f" | val_RMSE="
                    f"{val_metrics['rmse_degC']:.4f}C"
                    f" | val_MAE="
                    f"{val_metrics['mae_degC']:.4f}C"
                )

            message += f" | lr={current_lr:.2e}"

            print(message)

        # ----------------------------------------------------
        # Validation-based checkpoint selection
        # ----------------------------------------------------
        if val_metrics is not None:
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                patience = 0

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.output_dir,
                        "best_model.pth",
                    ),
                )

                best_info = {
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "best_val_rmse_degC": (
                        val_metrics["rmse_degC"]
                    ),
                    "best_val_mae_degC": (
                        val_metrics["mae_degC"]
                    ),
                    "selection_source": "validation_only",
                    "test_metrics_used_for_selection": False,
                }

                with open(
                    os.path.join(
                        args.output_dir,
                        "best_metrics.json",
                    ),
                    "w",
                ) as file:
                    json.dump(best_info, file, indent=2)

            else:
                patience += 1

            if patience >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best validation epoch={best_epoch}"
                )
                break

    # Ensure all final CUDA work has completed.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    training_validation_time_s = (
        time.perf_counter() - training_start
    )

    # ========================================================
    # Save last-epoch model and history
    # ========================================================
    torch.save(
        model.state_dict(),
        os.path.join(
            args.output_dir,
            "final_model.pth",
        ),
    )

    with open(
        os.path.join(
            args.output_dir,
            "history.json",
        ),
        "w",
    ) as file:
        json.dump(history, file, indent=2)

    print(
        "\nTraining and validation time: "
        f"{training_validation_time_s:.2f} s "
        f"({training_validation_time_s / 60.0:.2f} min)"
    )

    # ========================================================
    # Final held-out test evaluation
    #
    # The held-out condition is evaluated exactly once.
    # ========================================================
    final_test_metrics = None
    test_model_file = None
    final_test_time_s = None

    if X_test is not None:
        print("\n--- Final Held-Out Testing ---")

        best_model_path = os.path.join(
            args.output_dir,
            "best_model.pth",
        )

        if X_val is not None and os.path.exists(best_model_path):
            model.load_state_dict(
                torch.load(
                    best_model_path,
                    map_location=device,
                    weights_only=False,
                )
            )

            test_model_file = "best_model.pth"

            print(
                "Using validation-selected model "
                f"from epoch {best_epoch}"
            )
        else:
            test_model_file = "final_model.pth"

            print(
                "Using final_model.pth because no independent "
                "validation-selected checkpoint is available."
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        test_start = time.perf_counter()

        final_test_metrics = evaluate(
            model=model,
            X_eval=X_test,
            Y_eval=Y_test,
            mask=mask,
            dT_mean=fold_dT_mean,
            dT_std=fold_dT_std,
            batch_size=args.batch_size,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        final_test_time_s = (
            time.perf_counter() - test_start
        )

        print(
            f"Test RMSE: "
            f"{final_test_metrics['rmse_degC']:.4f} C"
        )
        print(
            f"Test MAE:  "
            f"{final_test_metrics['mae_degC']:.4f} C"
        )
        print(
            f"One-time held-out evaluation: "
            f"{final_test_time_s:.4f} s"
        )

        test_output = {
            "model_file_used": test_model_file,
            "best_validation_epoch": (
                best_epoch
                if X_val is not None
                else None
            ),
            "test_evaluated_during_training": False,
            "test_evaluation_count_after_training": 1,
            "test_loss": final_test_metrics["loss"],
            "test_rmse_degC": (
                final_test_metrics["rmse_degC"]
            ),
            "test_mae_degC": (
                final_test_metrics["mae_degC"]
            ),
            "final_test_time_s": final_test_time_s,
            "training_validation_time_s": (
                training_validation_time_s
            ),
            "training_validation_time_min": (
                training_validation_time_s / 60.0
            ),
            "normalization_source": (
                "training_samples_and_active_pixels_only"
            ),
            "fold_dT_mean": fold_dT_mean,
            "fold_dT_std": fold_dT_std,
            "device_info": get_device_info(device),
        }

        test_metrics_path = os.path.join(
            args.output_dir,
            "test_metrics.json",
        )

        with open(test_metrics_path, "w") as file:
            json.dump(test_output, file, indent=2)

        print(f"Saved test metrics: {test_metrics_path}")

    # ========================================================
    # Runtime output
    # ========================================================
    end_to_end_timed_procedure_s = (
        training_validation_time_s
        + (
            final_test_time_s
            if final_test_time_s is not None
            else 0.0
        )
    )

    runtime_output = {
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_completed,
        "early_stopping_active": X_val is not None,
        "best_validation_epoch": (
            best_epoch
            if X_val is not None
            else None
        ),
        "training_validation_time_s": (
            training_validation_time_s
        ),
        "training_validation_time_min": (
            training_validation_time_s / 60.0
        ),
        "final_test_time_s": final_test_time_s,
        "end_to_end_timed_procedure_s": (
            end_to_end_timed_procedure_s
        ),
        "end_to_end_timed_procedure_min": (
            end_to_end_timed_procedure_s / 60.0
        ),
        "test_evaluated_during_epochs": False,
        "device_info": get_device_info(device),
    }

    runtime_path = os.path.join(
        args.output_dir,
        "training_runtime.json",
    )

    with open(runtime_path, "w") as file:
        json.dump(runtime_output, file, indent=2)

    print(f"\nSaved runtime: {runtime_path}")
    print(
        "Complete timed procedure: "
        f"{end_to_end_timed_procedure_s:.2f} s "
        f"({end_to_end_timed_procedure_s / 60.0:.2f} min)"
    )
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()