"""
make_direct_ewma_dataset_from_raw.py

Build a direct-EWMA FNO dataset directly from raw ANSYS exports.

Raw files expected under --data_dir:
    coordinate_exp_test.txt
    temp_exporttest.txt
    NodeTemperature100.csv
    NodeTemperature87_5.csv
    NodeTemperature75_2.csv
    NodeTemperature62_5.csv
    NodeTemperature50.csv
    NodeTemperature25.csv

Output input channels:
    [u0, t_norm, x, y, Q_input, p, ewma_3, ewma_10, ewma_30]

Output target:
    dT(t) = T(t) - u0

Compatibility defaults reproduce the earlier two-stage pipeline:
    --heat_source_mode scaled
        Q_input = p * Q_base
    --ewma_alignment previous
        Store EWMA before updating it with the current load.
    --target_storage global_normalized
        Store globally standardized dT in Y_all.

For a cleaner future pipeline, use:
    --heat_source_mode base
    --ewma_alignment current
    --target_storage physical

WARNING:
Changing heat_source_mode or ewma_alignment changes the model input
semantics and requires retraining all affected models.

Example compatible build:
    python make_direct_ewma_dataset_from_raw.py \
        --data_dir data/raw \
        --output data/processed/fno_6cond_element_512_direct_ewma.h5 \
        --resolution 512 \
        --spans 3 10 30 \
        --heat_source_mode scaled \
        --ewma_alignment previous \
        --target_storage global_normalized
"""

import argparse
import json
import os
from collections import OrderedDict

import h5py
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


P_FULL_W = 988.0

CONDITIONS = OrderedDict([
    ("25", {
        "file": "NodeTemperature25.csv",
        "power_fraction": 0.25,
        "label": "25%",
    }),
    ("62.5", {
        "file": "NodeTemperature62_5.csv",
        "power_fraction": 0.625,
        "label": "62.5%",
    }),
    ("75", {
        "file": "NodeTemperature75_2.csv",
        "power_fraction": 0.75,
        "label": "75%",
    }),
    ("87.5", {
        "file": "NodeTemperature87_5.csv",
        "power_fraction": 0.875,
        "label": "87.5%",
    }),
    ("full", {
        "file": "NodeTemperature100.csv",
        "power_fraction": 1.0,
        "label": "Full (100%)",
    }),
    ("half", {
        "file": "NodeTemperature50.csv",
        "power_fraction": 0.5,
        "label": "Half (50%)",
    }),
])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a direct-EWMA FNO HDF5 dataset from raw ANSYS exports."
    )
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument(
        "--output",
        default="data/processed/fno_6cond_element_512_direct_ewma.h5",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--interp_method",
        choices=["linear", "nearest", "cubic"],
        default="linear",
    )
    parser.add_argument("--spans", type=int, nargs="+", default=[3, 10, 30])
    parser.add_argument(
        "--heat_source_mode",
        choices=["scaled", "base"],
        default="scaled",
        help=(
            "scaled: Q_input=p*Q_base, reproducing the old pipeline; "
            "base: Q_input=Q_base."
        ),
    )
    parser.add_argument(
        "--ewma_alignment",
        choices=["previous", "current"],
        default="previous",
        help=(
            "previous: store memory before current-load update; "
            "current: update with current load before storing."
        ),
    )
    parser.add_argument(
        "--target_storage",
        choices=["global_normalized", "physical"],
        default="global_normalized",
        help=(
            "global_normalized reproduces the old HDF5 contract; "
            "physical stores dT directly in degrees Celsius."
        ),
    )
    parser.add_argument(
        "--skip_initial",
        action="store_true",
        help="Do not store the initial-time sample; u0 remains the true initial field.",
    )
    parser.add_argument(
        "--require_all_conditions",
        action="store_true",
        help="Fail if any configured condition file is missing.",
    )
    parser.add_argument("--compression_level", type=int, default=4)
    return parser.parse_args()


def alpha_from_span(span):
    if span <= 0:
        raise ValueError(f"EWMA span must be positive, got {span}.")
    return 2.0 / (span + 1.0)


def load_coordinates(data_dir):
    path = os.path.join(data_dir, "coordinate_exp_test.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing coordinate file: {path}")

    print(f"  Loading coordinates: {path}")
    try:
        data = np.loadtxt(path, skiprows=1)
    except Exception:
        data = np.loadtxt(path, delimiter=",", skiprows=1)

    data = np.atleast_2d(data)
    if data.shape[1] >= 3:
        node_ids = data[:, 0].astype(np.int64)
        positions = data[:, 1:3].astype(np.float64)
    elif data.shape[1] == 2:
        node_ids = np.arange(1, len(data) + 1, dtype=np.int64)
        positions = data[:, :2].astype(np.float64)
    else:
        raise ValueError(f"Coordinate file must contain at least two columns: {path}")

    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("Coordinate file contains duplicate node IDs.")

    print(f"    {len(positions)} nodes")
    return node_ids, positions, path


def load_heat_source(data_dir):
    path = os.path.join(data_dir, "temp_exporttest.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing heat-source file: {path}")

    print(f"  Loading heat source: {path}")
    centroids = []
    heat_values = []

    with open(path, "r", encoding="utf-8", errors="replace") as file:
        next(file, None)
        for line_number, line in enumerate(file, start=2):
            parts = [part.strip() for part in line.strip().split("\t") if part.strip()]
            if len(parts) < 5:
                continue
            try:
                centroids.append([float(parts[1]), float(parts[2])])
                heat_values.append(float(parts[4]))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid heat-source row at {path}:{line_number}: {line.strip()}"
                ) from exc

    if not heat_values:
        raise ValueError(f"No valid heat-source rows were read from {path}")

    centroids = np.asarray(centroids, dtype=np.float64)
    heat_values = np.asarray(heat_values, dtype=np.float32)
    print(f"    {len(heat_values)} elements, heated={(heat_values > 0).sum()}")
    return centroids, heat_values, path


def load_node_temperature(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    if data.shape[1] < 3:
        raise ValueError(
            f"Temperature file must contain time, node ID, and temperature: {path}"
        )

    all_times = data[:, 0].astype(np.float64)
    all_nodes = data[:, 1].astype(np.int64)
    all_temperatures = data[:, 2].astype(np.float32)

    unique_times = np.unique(all_times)
    unique_nodes = np.unique(all_nodes)
    if len(unique_times) == 0 or len(unique_nodes) == 0:
        raise ValueError(f"No temperature records found in {path}")

    node_to_row = {node_id: i for i, node_id in enumerate(unique_nodes)}
    T_fields = np.full(
        (len(unique_nodes), len(unique_times)),
        np.nan,
        dtype=np.float32,
    )

    for time_value, node_id, temperature in zip(
        all_times, all_nodes, all_temperatures
    ):
        time_index = int(np.searchsorted(unique_times, time_value))
        node_index = node_to_row[int(node_id)]
        T_fields[node_index, time_index] = temperature

    if np.isnan(T_fields).any():
        missing_count = int(np.isnan(T_fields).sum())
        raise ValueError(
            f"Temperature matrix from {path} has {missing_count} missing node-time values."
        )

    print(
        f"    {os.path.basename(path)}: {len(unique_nodes)} nodes x "
        f"{len(unique_times)} timesteps, "
        f"t=[{unique_times[0]:.1f}, {unique_times[-1]:.1f}] s, "
        f"T=[{T_fields.min():.1f}, {T_fields.max():.1f}] degC"
    )
    return T_fields, unique_times, unique_nodes


def create_regular_grid(positions, resolution):
    if resolution < 2:
        raise ValueError("Resolution must be at least 2.")

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    dx = (x_max - x_min) * 0.02
    dy = (y_max - y_min) * 0.02

    x_grid = np.linspace(x_min - dx, x_max + dx, resolution, dtype=np.float64)
    y_grid = np.linspace(y_min - dy, y_max + dy, resolution, dtype=np.float64)
    xx, yy = np.meshgrid(x_grid, y_grid)
    return x_grid, y_grid, xx, yy


def create_geometry_mask(positions, xx, yy, threshold=None):
    tree = cKDTree(positions)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    distances, _ = tree.query(grid_points)

    if threshold is None:
        x_spacing = abs(float(xx[0, 1] - xx[0, 0]))
        y_spacing = abs(float(yy[1, 0] - yy[0, 0]))
        threshold = max(x_spacing, y_spacing) * 1.5

    return (distances < threshold).reshape(xx.shape).astype(np.float32)


def rasterize_field(positions, values, xx, yy, mask, method):
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    grid_field = griddata(positions, values, grid_points, method=method).reshape(xx.shape)

    if np.any(np.isnan(grid_field)):
        nearest = griddata(
            positions, values, grid_points, method="nearest"
        ).reshape(xx.shape)
        grid_field = np.where(np.isnan(grid_field), nearest, grid_field)

    return (grid_field * mask).astype(np.float32)


def match_temperature_nodes(coordinate_ids, coordinate_positions, temperature_ids, T_fields):
    id_to_position_index = {
        int(node_id): i for i, node_id in enumerate(coordinate_ids)
    }
    matched_temperature_rows = []
    matched_position_rows = []
    missing_ids = []

    for temperature_row, node_id in enumerate(temperature_ids):
        position_index = id_to_position_index.get(int(node_id))
        if position_index is None:
            missing_ids.append(int(node_id))
        else:
            matched_temperature_rows.append(temperature_row)
            matched_position_rows.append(position_index)

    if not matched_temperature_rows:
        raise ValueError("No temperature nodes matched the coordinate file.")

    if missing_ids:
        print(
            f"    WARNING: {len(missing_ids)} temperature nodes were not found "
            "in the coordinate file."
        )

    matched_positions = coordinate_positions[np.asarray(matched_position_rows)]
    matched_T_fields = T_fields[np.asarray(matched_temperature_rows)]
    return matched_positions, matched_T_fields


def rasterize_temperature_sequence(
    coordinate_ids,
    coordinate_positions,
    temperature_ids,
    T_fields,
    xx,
    yy,
    mask,
    method,
):
    positions, matched_fields = match_temperature_nodes(
        coordinate_ids,
        coordinate_positions,
        temperature_ids,
        T_fields,
    )

    n_times = matched_fields.shape[1]
    grids = np.empty((n_times, *mask.shape), dtype=np.float32)
    print(f"    Rasterizing {n_times} timesteps...", end=" ", flush=True)
    for time_index in range(n_times):
        grids[time_index] = rasterize_field(
            positions,
            matched_fields[:, time_index],
            xx,
            yy,
            mask,
            method,
        )
    print("done")
    return grids


def compute_ewma_features(n_samples, load_fraction, spans, alignment):
    memories = np.zeros(len(spans), dtype=np.float32)
    result = np.empty((n_samples, len(spans)), dtype=np.float32)

    for sample_index in range(n_samples):
        if alignment == "current":
            for j, span in enumerate(spans):
                alpha = alpha_from_span(span)
                memories[j] = alpha * load_fraction + (1.0 - alpha) * memories[j]
            result[sample_index] = memories
        else:
            result[sample_index] = memories
            for j, span in enumerate(spans):
                alpha = alpha_from_span(span)
                memories[j] = alpha * load_fraction + (1.0 - alpha) * memories[j]

    return result


def update_running_stats(running, values):
    values64 = np.asarray(values, dtype=np.float64)
    running["sum"] += float(values64.sum())
    running["sum_sq"] += float(np.square(values64).sum())
    running["count"] += int(values64.size)


def finalize_running_stats(running, name):
    if running["count"] == 0:
        raise ValueError(f"No values were available to compute {name} statistics.")
    mean = running["sum"] / running["count"]
    variance = running["sum_sq"] / running["count"] - mean**2
    std = max(float(max(variance, 0.0) ** 0.5), 1e-8)
    return float(mean), std


def scan_available_conditions(data_dir, require_all):
    available = []
    missing = []
    for key, info in CONDITIONS.items():
        path = os.path.join(data_dir, info["file"])
        if os.path.exists(path):
            available.append((key, info, path))
        else:
            missing.append(path)

    if require_all and missing:
        raise FileNotFoundError(
            "Required condition files are missing:\n  " + "\n  ".join(missing)
        )
    if not available:
        raise FileNotFoundError(
            f"No configured temperature files were found under {data_dir}."
        )
    for path in missing:
        print(f"  SKIPPED missing condition file: {path}")
    return available


def main():
    args = parse_args()

    if len(set(args.spans)) != len(args.spans):
        raise ValueError("EWMA spans must be unique.")
    for span in args.spans:
        alpha_from_span(span)
    if not 0 <= args.compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9.")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("RAW ANSYS -> DIRECT EWMA FNO DATASET")
    print("=" * 80)
    print(f"Data directory:      {args.data_dir}")
    print(f"Output:              {args.output}")
    print(f"Resolution:          {args.resolution} x {args.resolution}")
    print(f"Interpolation:       {args.interp_method}")
    print(f"EWMA spans:          {args.spans}")
    print(f"EWMA alignment:      {args.ewma_alignment}")
    print(f"Heat-source mode:    {args.heat_source_mode}")
    print(f"Target storage:      {args.target_storage}")
    print(f"Skip initial sample: {args.skip_initial}")
    print()

    print("--- Loading geometry and heat source ---")
    node_ids, positions, coordinate_path = load_coordinates(args.data_dir)
    heat_centroids, heat_values, heat_path = load_heat_source(args.data_dir)

    print("\n--- Creating common grid and geometry mask ---")
    x_grid, y_grid, xx, yy = create_regular_grid(positions, args.resolution)
    mask = create_geometry_mask(positions, xx, yy)
    mask_bool = mask > 0
    active_pixels = int(mask_bool.sum())
    coverage_percent = active_pixels / mask.size * 100.0
    print(f"  Active pixels: {active_pixels} ({coverage_percent:.2f}% coverage)")

    print("\n--- Rasterizing base heat source ---")
    Q_grid_base = rasterize_field(
        heat_centroids,
        heat_values,
        xx,
        yy,
        mask,
        args.interp_method,
    )
    print(f"  Q_base range: [{Q_grid_base[mask_bool].min():.6g}, "
          f"{Q_grid_base[mask_bool].max():.6g}]")

    available_conditions = scan_available_conditions(
        args.data_dir, args.require_all_conditions
    )

    # First pass: rasterize each condition, cache its physical fields, and
    # compute the global statistics needed to reproduce the old HDF5 contract.
    print("\n--- First pass: rasterization and statistics ---")
    condition_cache = OrderedDict()
    u0_stats = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    dT_stats = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    Q_stats = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    source_files = [coordinate_path, heat_path]
    total_samples = 0
    global_time_min = None
    global_time_max = None

    for condition_key, condition_info, temperature_path in available_conditions:
        print(f"\n  [{condition_key}] {condition_info['label']}")
        T_fields, times_sec, temperature_ids = load_node_temperature(temperature_path)
        T_grids = rasterize_temperature_sequence(
            node_ids,
            positions,
            temperature_ids,
            T_fields,
            xx,
            yy,
            mask,
            args.interp_method,
        )

        initial_field = T_grids[0].copy()
        elapsed_sec = times_sec - times_sec[0]
        elapsed_max = max(float(elapsed_sec[-1]), 1e-8)
        time_norm = (elapsed_sec / elapsed_max).astype(np.float32)

        selected = np.arange(len(times_sec), dtype=np.int64)
        if args.skip_initial:
            selected = selected[1:]
        if len(selected) == 0:
            raise ValueError(
                f"Condition {condition_key} contains no stored samples after filtering."
            )

        p = float(condition_info["power_fraction"])
        ewma_all = compute_ewma_features(
            len(times_sec), p, args.spans, args.ewma_alignment
        )
        Q_input = Q_grid_base * p if args.heat_source_mode == "scaled" else Q_grid_base

        # Match the old preprocessing statistics exactly: u0 and Q are counted
        # once for every stored sample; dT includes every stored sample/pixel.
        n_stored = len(selected)
        for _ in range(n_stored):
            update_running_stats(u0_stats, initial_field[mask_bool])
            q_nonzero = Q_input[mask_bool]
            q_nonzero = q_nonzero[q_nonzero > 0]
            if q_nonzero.size:
                update_running_stats(Q_stats, q_nonzero)

        for time_index in selected:
            dT = T_grids[time_index] - initial_field
            update_running_stats(dT_stats, dT[mask_bool])

        condition_cache[condition_key] = {
            "info": condition_info,
            "source_path": temperature_path,
            "T_grids": T_grids,
            "times_sec": times_sec.astype(np.float64),
            "elapsed_sec": elapsed_sec.astype(np.float32),
            "time_norm": time_norm,
            "selected": selected,
            "u0": initial_field,
            "ewma": ewma_all,
            "Q_input": Q_input.astype(np.float32),
        }
        source_files.append(temperature_path)
        total_samples += n_stored
        global_time_min = (
            float(times_sec.min())
            if global_time_min is None
            else min(global_time_min, float(times_sec.min()))
        )
        global_time_max = (
            float(times_sec.max())
            if global_time_max is None
            else max(global_time_max, float(times_sec.max()))
        )
        print(f"    -> {n_stored} stored samples, p={p:.3f}")

    T_abs_mean, T_abs_std = finalize_running_stats(u0_stats, "u0")
    dT_mean, dT_std = finalize_running_stats(dT_stats, "dT")
    if Q_stats["count"]:
        Q_mean, Q_std = finalize_running_stats(Q_stats, "Q")
    else:
        Q_mean, Q_std = 0.0, 1.0

    print("\n--- Global preprocessing statistics ---")
    print(f"  T_abs_mean/std: {T_abs_mean:.6f} / {T_abs_std:.6f} degC")
    print(f"  dT_mean/std:    {dT_mean:.6f} / {dT_std:.6f} degC")
    print(f"  Q_mean/std:     {Q_mean:.6f} / {Q_std:.6f}")
    print(f"  Total samples:  {total_samples}")

    n_channels = 6 + len(args.spans)
    height, width = mask.shape
    xx_norm = np.linspace(0, 1, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    yy_norm = np.linspace(0, 1, height, dtype=np.float32)[:, None].repeat(width, axis=1)

    print(f"\n--- Writing HDF5: {args.output} ---")
    with h5py.File(args.output, "w") as output:
        X_dataset = output.create_dataset(
            "X_all",
            shape=(total_samples, n_channels, height, width),
            dtype=np.float32,
            chunks=(1, n_channels, height, width),
            compression="gzip",
            compression_opts=args.compression_level,
        )
        Y_dataset = output.create_dataset(
            "Y_all",
            shape=(total_samples, 1, height, width),
            dtype=np.float32,
            chunks=(1, 1, height, width),
            compression="gzip",
            compression_opts=args.compression_level,
        )
        power_dataset = output.create_dataset(
            "power_fraction_all", shape=(total_samples,), dtype=np.float32
        )
        time_norm_dataset = output.create_dataset(
            "time_norm_all", shape=(total_samples,), dtype=np.float32
        )
        elapsed_dataset = output.create_dataset(
            "elapsed_time_all", shape=(total_samples,), dtype=np.float32
        )

        output.create_dataset("mask", data=mask.astype(np.float32))
        output.create_dataset("x_grid", data=x_grid)
        output.create_dataset("y_grid", data=y_grid)
        output.create_dataset("Q_grid_base_physical", data=Q_grid_base)

        cursor = 0
        sample_counts = {}
        total_powers = {}

        for condition_key, cached in condition_cache.items():
            condition_info = cached["info"]
            p = float(condition_info["power_fraction"])
            condition_start = cursor
            u0_norm = ((cached["u0"] - T_abs_mean) / T_abs_std) * mask
            Q_norm = ((cached["Q_input"] - Q_mean) / Q_std) * mask

            for time_index in cached["selected"]:
                t_norm_value = float(cached["time_norm"][time_index])
                t_channel = np.full((height, width), t_norm_value, dtype=np.float32)
                p_channel = np.full((height, width), p, dtype=np.float32)
                channels = [
                    u0_norm.astype(np.float32),
                    t_channel,
                    xx_norm,
                    yy_norm,
                    Q_norm.astype(np.float32),
                    p_channel,
                ]
                for memory_value in cached["ewma"][time_index]:
                    channels.append(
                        np.full(
                            (height, width),
                            float(memory_value),
                            dtype=np.float32,
                        )
                    )

                dT_physical = cached["T_grids"][time_index] - cached["u0"]
                if args.target_storage == "global_normalized":
                    target = ((dT_physical - dT_mean) / dT_std) * mask
                else:
                    target = dT_physical * mask

                X_dataset[cursor] = np.stack(channels, axis=0).astype(np.float32)
                Y_dataset[cursor, 0] = target.astype(np.float32)
                power_dataset[cursor] = p
                time_norm_dataset[cursor] = t_norm_value
                elapsed_dataset[cursor] = float(cached["elapsed_sec"][time_index])
                cursor += 1

            indices = np.arange(condition_start, cursor, dtype=np.int64)
            output.create_dataset(f"idx_{condition_key}", data=indices)
            sample_counts[condition_key] = int(len(indices))
            total_powers[condition_key] = P_FULL_W * p

        if cursor != total_samples:
            raise RuntimeError(
                f"Internal sample-count mismatch: wrote {cursor}, expected {total_samples}."
            )

        # Normalization and physical metadata.
        output.attrs["T_abs_mean"] = T_abs_mean
        output.attrs["T_abs_std"] = T_abs_std
        output.attrs["Q_mean"] = Q_mean
        output.attrs["Q_std"] = Q_std
        output.attrs["dT_mean"] = dT_mean
        output.attrs["dT_std"] = dT_std
        output.attrs["target_storage"] = args.target_storage
        output.attrs["target_is_normalized"] = (
            args.target_storage == "global_normalized"
        )

        # Dataset definition.
        output.attrs["dataset_type"] = "direct_ewma_from_raw"
        output.attrs["input_type"] = "direct_ewma"
        output.attrs["target_type"] = "dT"
        output.attrs["input_definition"] = (
            "[u0, t_norm, x, y, Q_input, p, "
            + ", ".join(f"ewma_{span}" for span in args.spans)
            + "]"
        )
        output.attrs["target_definition"] = "dT = T(t) - u0"
        output.attrs["heat_source_mode"] = args.heat_source_mode
        output.attrs["Q_input_definition"] = (
            "p * Q_base"
            if args.heat_source_mode == "scaled"
            else "Q_base"
        )
        output.attrs["ewma_alignment"] = args.ewma_alignment
        output.attrs["ewma_initial_state"] = "zero"
        output.attrs["ewma_update_equation"] = (
            "m_i = alpha*p_i + (1-alpha)*m_previous"
        )
        output.attrs["ewma_spans_json"] = json.dumps(args.spans)
        output.attrs["ewma_alphas_json"] = json.dumps(
            [alpha_from_span(span) for span in args.spans]
        )
        output.attrs["skip_initial"] = args.skip_initial

        # Grid and source metadata.
        output.attrs["resolution"] = args.resolution
        output.attrs["interp_method"] = args.interp_method
        output.attrs["P_full"] = P_FULL_W
        output.attrs["n_nodes_original"] = len(node_ids)
        output.attrs["n_elements"] = len(heat_values)
        output.attrs["coverage_percent"] = coverage_percent
        output.attrs["t_min"] = global_time_min
        output.attrs["t_max"] = global_time_max
        output.attrs["in_channels"] = n_channels
        output.attrs["n_input_channels"] = n_channels
        output.attrs["n_output_channels"] = 1
        output.attrs["source_files_json"] = json.dumps(source_files)
        output.attrs["conditions_json"] = json.dumps(list(condition_cache.keys()))
        output.attrs["n_conditions"] = len(condition_cache)

        for condition_key in condition_cache:
            output.attrs[f"total_power_{condition_key}"] = total_powers[condition_key]
            output.attrs[f"n_{condition_key}"] = sample_counts[condition_key]
            output.attrs[f"power_fraction_{condition_key}"] = float(
                condition_cache[condition_key]["info"]["power_fraction"]
            )
            output.attrs[f"source_file_{condition_key}"] = condition_cache[
                condition_key
            ]["source_path"]

    print("\n" + "=" * 80)
    print("DIRECT EWMA DATASET CREATION COMPLETE")
    print("=" * 80)
    print(f"Output: {args.output}")
    print(f"X_all shape: ({total_samples}, {n_channels}, {height}, {width})")
    print(f"Y_all shape: ({total_samples}, 1, {height}, {width})")
    print("Samples per condition:")
    for condition_key, count in sample_counts.items():
        print(f"  {condition_key:>6}: {count}")
    print(f"  {'TOTAL':>6}: {sum(sample_counts.values())}")
    print("\nReproducibility-critical settings:")
    print(f"  heat_source_mode = {args.heat_source_mode}")
    print(f"  ewma_alignment   = {args.ewma_alignment}")
    print(f"  target_storage   = {args.target_storage}")


if __name__ == "__main__":
    main()