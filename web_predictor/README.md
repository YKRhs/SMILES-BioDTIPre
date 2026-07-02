# Web predictor

This folder contains the Flask web interface for SMILES-BioDTIPre.

## Required files

Place these files or directories in the repository root before starting the web app:

```text
smilesT5/
bioT5+/
best_model_xgb.pkl
best_model_lgb.pkl
configs/ensemble_config_300_search_mcc.json
```

## Start the app

```bash
python app.py
```

On Windows, you can also run:

```bash
start_web.bat
```

`start_web.bat` changes the working directory to this folder, runs `python app.py`, and keeps the command window open.

Open the web page at:

```text
http://127.0.0.1:5000
```

## Input format

Single prediction requires one SMILES string and one protein sequence.

Batch prediction accepts a CSV file. Supported column names are:

```text
SMILES, Protein
SMILES, Target Sequence
SMILES, Sequence
```

The output contains XGBoost probability, LightGBM probability, ensemble probability, and the final predicted class.
