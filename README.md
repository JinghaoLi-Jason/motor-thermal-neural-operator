# Direct EWMA Fourier Neural Operator for Motor Thermal Prediction

This repository contains the data-processing, training, evaluation, and
visualization workflow for a Fourier Neural Operator (FNO) that predicts
the transient temperature field of an electric motor.

The final approach is a **direct FNO with exponentially weighted moving
average (EWMA) load-memory features**. Each requested time point is
predicted directly.

## Model formulation

The model input at time index \(i\) is

\[
\mathbf{a}(\mathbf{x},t_i)
=
\left[
u_0(\mathbf{x}),
t_{\mathrm{norm},i},
x,
y,
Q_i(\mathbf{x}),
p_i,
m_i^{(3)},
m_i^{(10)},
m_i^{(30)}
\right],
\]

where

\[
Q_i(\mathbf{x})
=
p_i Q_{\mathrm{base}}(\mathbf{x}).
\]

The channels are:

1. `u0`: normalized initial temperature field
2. `t_norm`: normalized elapsed time
3. `x`: normalized spatial coordinate
4. `y`: normalized spatial coordinate
5. `Q_input`: load-scaled heat-source field \(p_iQ_{\mathrm{base}}\)
6. `p`: current load fraction
7. `ewma_3`: short-term load-memory feature
8. `ewma_10`: medium-term load-memory feature
9. `ewma_30`: long-term load-memory feature

The output is the temperature rise

\[
\Delta T(\mathbf{x},t_i)
=
T(\mathbf{x},t_i)-u_0(\mathbf{x}).
\]

The absolute temperature field is reconstructed as

\[
T(\mathbf{x},t_i)
=
u_0(\mathbf{x})+\Delta T(\mathbf{x},t_i).
\]

## EWMA definition

For the dataset used by the reported model, the EWMA state is initialized
to zero and the stored feature at sample \(i\) represents the load history
before the current sample:

\[
m_0^{(s)}=0,
\]

\[
m_i^{(s)}
=
\alpha_s p_{i-1}
+
(1-\alpha_s)m_{i-1}^{(s)},
\qquad i\geq1,
\]

where

\[
\alpha_s=\frac{2}{s+1}.
\]

The current load \(p_i\) is supplied separately as an input channel. After
sample \(i\) is constructed, \(p_i\) updates the EWMA state used by the
next sample.

The first stored EWMA values are therefore zero.

## Repository structure

```text
train/direct_ewma/
├── make_direct_ewma_dataset.py
├── visualize_dataset.py
├── plot_dataset_ewma_features.py
├── train_direct_ewma.py
├── eval_direct_ewma_cv.py
└── infer_direct_ewma_varying.py
```

### Script descriptions

#### `make_direct_ewma_dataset.py`

Creates the direct-EWMA HDF5 dataset from the raw ANSYS temperature,
coordinate, and heat-source exports.

The generated dataset contains:

```text
X_all: [N, 9, H, W]
Y_all: [N, 1, H, W]
mask: [H, W]
x_grid: [W]
y_grid: [H]
power_fraction_all: [N]
time_norm_all: [N]
elapsed_time_all: [N]
idx_<condition>: sample indices for each condition
```

#### `visualize_dataset.py`

Inspects the processed HDF5 dataset and generates:

- Spatial input-field visualizations
- Temperature evolution curves
- Representative temperature fields
- Target distributions
- Condition-level summaries
- Sample-level summaries

#### `plot_dataset_ewma_features.py`

Plots the load fraction and EWMA memory features for the processed
operating conditions.

#### `train_direct_ewma.py`

Trains the direct EWMA-FNO.

Supported scenarios include:

- `final_all`: train a final model using all selected conditions
- `train_val`: divide selected conditions into training and validation
- `loco_trainall`: leave one condition out for testing without validation
- `loco_val`: leave one condition out for testing and use an independent
  validation subset for model selection

For clean cross-validation, `loco_val` is recommended. Validation controls
checkpoint selection and early stopping. The held-out condition should be
evaluated only after model selection.

#### `eval_direct_ewma_cv.py`

Evaluates leave-one-condition-out cross-validation folds. It generates:

- Field RMSE and MAE
- Maximum-temperature RMSE and MAE
- Mean-temperature RMSE and MAE
- Temperature time-series comparisons
- Representative field comparisons
- Point-wise temperature traces
- Global cross-validation summaries

#### `infer_direct_ewma_varying.py`

Evaluates the final model on the varying-load profile

```text
100% -> 50% -> 75%
```

with switch times:

```text
19000 s
39000 s
```

and a final simulation time of:

```text
60000 s
```

It generates:

- Load and EWMA curves
- Global maximum- and mean-temperature comparisons
- Error curves
- Temperature-field comparisons
- Point-wise temperature traces
- CSV and JSON metrics
- Full-profile inference runtime

## Raw data

The preprocessing workflow expects the following files under `data/raw/`:

```text
coordinate_exp_test.txt
temp_exporttest.txt
NodeTemperature25.csv
NodeTemperature50.csv
NodeTemperature62_5.csv
NodeTemperature75_2.csv
NodeTemperature87_5.csv
NodeTemperature100.csv
```

The constant-load temperature files use the format:

```text
time, node_id, temperature
```

The coordinate file contains node identifiers and spatial coordinates.

The heat-source file contains element centroids and heat-source values.

Raw and processed data are not stored in the Git repository. Access is
subject to the applicable company data-management requirements.

## Installation

Create and activate a Python environment before installing the
dependencies.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For HPC use, activate the appropriate CUDA, Python, and environment
modules before installing or running the code.

## Dataset creation

To reproduce the dataset used by the reported model:

```bash
python train/direct_ewma/make_direct_ewma_dataset.py \
    --data_dir data/raw \
    --output data/processed/fno_6cond_element_512_direct_ewma.h5 \
    --resolution 512 \
    --spans 3 10 30 \
    --heat_source_mode scaled \
    --ewma_alignment previous \
    --target_storage global_normalized \
    --require_all_conditions
```

The reproducibility-critical settings are:

```text
heat_source_mode = base
ewma_alignment   = previous
target_storage   = global_normalized
```

## Dataset visualization

```bash
python train/direct_ewma/visualize_dataset.py \
    --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 \
    --output_dir results/dataset_visualization \
    --dpi 600
```

## Final-model training

The main final model uses the constant-load conditions:

```text
50%, 75%, 100%
```

Example:

```bash
python train/direct_ewma/train_direct_ewma.py \
    --scenario final_all \
    --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 \
    --output_dir results/direct_ewma_model/final_50_75_100 \
    --include_conditions half,75,full \
    --device cuda:1
```

## Leave-one-condition-out cross-validation

Example for a held-out 25% condition:

```bash
python train/direct_ewma/train_direct_ewma.py \
    --scenario loco_val \
    --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 \
    --output_dir results/direct_ewma_cv/test_25 \
    --include_conditions 25,half,75,full \
    --test_condition 25 \
    --device cuda:1
```

Equivalent runs should be performed for:

```text
25%
50%
75%
100%
```

The fold-specific model normalization parameters are calculated from
training samples only and saved in each fold's `config.json`.

## Cross-validation evaluation

```bash
python train/direct_ewma/eval_direct_ewma_cv.py \
    --cv_root results/direct_ewma_cv \
    --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 \
    --device cuda:1 \
    --model_file best_model.pth
```

For `loco_val`, `best_model.pth` is the validation-selected checkpoint.

## Varying-load inference

```bash
python train/direct_ewma/infer_direct_ewma_varying.py \
    --model_dir results/direct_ewma_model/final_all \
    --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 \
    --val_csv data/raw/NodeTemperature_100_50_75_correct_sampling_time.csv \
    --coord_file data/raw/coordinate_exp_test.txt \
    --output_dir results/final_inference_3cond \
    --device cuda:1 \
    --t_end 60000
```

The inference procedure must use the same EWMA alignment as the training
dataset:

1. Use the existing EWMA state for the current prediction.
2. Predict the current time point.
3. Update the EWMA state using the current load.
4. Proceed to the next time point.

## Model configuration

The main FNO configuration is:

```text
Input channels:          9
Output channels:         1
Fourier layers:          4
Hidden-channel width:    32
Retained Fourier modes:  16 x 16
Optimizer:               AdamW
Initial learning rate:   1e-3
Weight decay:            1e-4
Batch size:              8
Maximum epochs:          100
Scheduler:               cosine annealing
Gradient clipping:       1.0
Loss:                    masked MSE
Spatial resolution:      512 x 512
```

## Loss function

Training minimizes masked mean-squared error over the active motor
geometry:

\[
\mathcal{L}
=
\frac{
\sum_{j=1}^{N}
\sum_{\mathbf{x}}
M(\mathbf{x})
\left[
\hat{u}_j(\mathbf{x})
-u_j(\mathbf{x})
\right]^2
}{
N\sum_{\mathbf{x}}M(\mathbf{x})
}.
\]

Pixels outside the geometry mask do not contribute to the loss.

## Normalization

The HDF5 dataset preserves global normalization parameters for
compatibility with the original workflow.

For leave-one-condition-out cross-validation, the training script
reconstructs physical temperature rise and calculates new normalization
statistics using training samples and active motor pixels only. The same
training-derived normalization is then applied to validation and held-out
test samples.

Each trained model stores its normalization parameters in `config.json`.
Inference scripts must use the model-specific values from `config.json`
when converting predictions back to physical units.

## Computational environment

The final experiments were performed using an NVIDIA A100 80 GB PCIe GPU.

Training runtime and inference runtime are stored in JSON files within the
corresponding result directories.

Inference runtime excludes:

- Model loading
- Raw FEM data loading
- Ground-truth rasterization
- Plot generation
- CSV and JSON writing

It represents the full-profile neural-operator prediction procedure,
including feature construction, model forward passes, output
denormalization, and temperature-summary extraction.

## Generated files

The training and evaluation workflows may generate:

```text
config.json
normalization.json
history.json
best_metrics.json
test_metrics.json
training_runtime.json
metrics.json
cv_inference_metrics.json
cv_summary.csv
cv_summary.json
final_model.pth
best_model.pth
```

Large datasets, checkpoints, caches, and generated result directories
should not be committed to the source-code repository unless explicitly
required by the project owner.

## Reproducibility notes

The standalone raw-data builder was validated against the original
two-stage preprocessing workflow.

Using absolute and relative tolerances of \(10^{-5}\):

- All nine model-input channels agreed.
- The active-domain temperature-rise target agreed.
- Geometry masks and spatial grids matched.
- Condition indices and sample counts matched.
- Load and EWMA features matched exactly.

The new dataset contains additional traceability arrays and metadata.
Inactive target pixels are explicitly stored as zero, whereas the legacy
dataset retained arbitrary normalized values outside the geometry. This
difference does not affect training or evaluation because the geometry
mask is applied to all error calculations.

## Data and confidentiality

This repository contains source code for the project workflow. Raw FEM
data, processed datasets, trained checkpoints, and company-specific
results must be stored and shared only through approved company storage
and according to the applicable project agreements.
