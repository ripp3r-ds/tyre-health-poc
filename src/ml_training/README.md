# ML Training

Training code for TireCheck AI (condition and inflation models) using PyTorch and MLflow.

## Environment

Option A — pip (CPU):
```bash
python -m venv .venv && . .venv/Scripts/activate  # on Windows
pip install -r requirements.txt
```

Option B — Azure ML job environment:
- See `aml/environment/conda_cpu.yaml` and `aml/environment/environment.yaml`.

## Datasets
- Condition (tread): classes `good`, `worn`
- Pressure: classes `full`, `flat`
Place train/val/test splits under `data/raw/<task>/<split>/<class>/` (see `data/`).

## Scripts
- `train_condition.py` — trains condition classifier
- `train_pressure.py` — trains pressure classifier
- `train_pressure_kfold.py` — k-fold CV for pressure

Common helpers:
- `dataloaders.py`, `model.py`, `utils.py`

## Run
```bash
cd src/ml_training
python train_condition.py  # or train_pressure.py
```

MLflow runs will be stored under `mlruns/`. Best checkpoints can be copied to `notebooks/models/` or registered to AML.

## Exporting models to Azure ML
Package a scoring script and environment from `score/` and deploy via AML online endpoints. The API calls AML using env vars `AML_ENDPOINT` and `AML_KEY`.


