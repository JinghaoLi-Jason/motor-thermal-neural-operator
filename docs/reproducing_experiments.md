# Reproducing the experiments

The source scripts are unchanged. The published checkpoint and experiment records support inspection, but full training and inference require the original FEM inputs or a dataset prepared with the same schema and semantics.

## Environment

Use Python 3.11 and install `requirements.txt`, or create the environment with:

```sh
conda env create -f environment.yml
conda activate motor-thermal-fno
```

The recorded training environment used PyTorch 2.5.1 with CUDA 12.1 and an NVIDIA A100. The portable dependency files omit machine-specific paths and the CUDA wheel suffix. Choose a suitable PyTorch build for the host. Recorded runtime should not be assumed to transfer to other hardware.

## Prepare data

Place coordinate, element heat-source, and nodal-temperature exports under `data/raw/`. Exact expected filenames are listed in the header of `train/direct_ewma/make_direct_ewma_dataset.py`.

For the input semantics recorded by the archived final varying-load evaluation:

```sh
python train/direct_ewma/make_direct_ewma_dataset.py --data_dir data/raw --output data/processed/fno_6cond_element_512_direct_ewma.h5 --resolution 512 --spans 3 10 30 --heat_source_mode base --ewma_alignment previous --target_storage global_normalized --require_all_conditions
```

The builder's `scaled` default belongs to an alternative input representation. Do not mix representations when reusing checkpoints. Inspect a local dataset's heat-source channel with:

```sh
python tools/check_dataset.py
```

This diagnostic prints metadata and channel differences across conditions. It is not a replacement for a full reproduction check.

## Train a final model

The bundled final checkpoint's configuration records five training conditions and 500 samples:

```sh
python train/direct_ewma/train_direct_ewma.py --scenario final_all --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 --output_dir results/final_five_conditions --include_conditions half,75,full,62.5,87.5 --device cpu
```

On a suitable CUDA host, replace `cpu` with an available device such as `cuda:0`. Full-resolution CPU training can be slow.

## Evaluate the published checkpoint

The varying-load procedure additionally requires the separate 100% → 50% → 75% FEM sequence and coordinate file:

```sh
python train/direct_ewma/infer_direct_ewma_varying.py --model_dir artifacts/pretrained/final_all --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 --val_csv data/raw/NodeTemperature_100_50_75_correct_sampling_time.csv --coord_file data/raw/coordinate_exp_test.txt --output_dir results/varying_load_reproduction --device cpu --t_end 60000
```

The original inference script loads its reference spatial channels from the HDF5 file and requires previous-aligned EWMA metadata. Match these inputs to the archived model configuration before interpreting results.

## Run held-out-condition experiments

For a new experiment with 25% held out and validation-based checkpoint selection:

```sh
python train/direct_ewma/train_direct_ewma.py --scenario loco_val --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 --output_dir results/direct_ewma_cv/test_25 --include_conditions 25,half,75,full --test_condition 25 --device cpu
python train/direct_ewma/eval_direct_ewma_cv.py --cv_root results/direct_ewma_cv --dataset data/processed/fno_6cond_element_512_direct_ewma.h5 --model_file best_model.pth --device cpu
```

Repeat with the other held-out conditions as needed. These commands describe new validation-selected experiments. The included historical CV summaries evaluated `final_model.pth`, as recorded in each fold's metrics, and should be interpreted accordingly.

## Inspect results without the full data

```sh
python examples/inspect_results.py
```

This standard-library command verifies checksums and consistency of the published artifact records. Full training, model inference, and a clean installation of the reduced scientific environment have not been rerun as part of packaging this repository.
