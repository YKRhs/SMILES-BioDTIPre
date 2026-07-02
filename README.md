# SMILES-BioDTIPre: SMILES-T5 and BioT5+ based DTI prediction

This repository contains the code for SMILES-BioDTIPre, a drug-target interaction prediction framework based on SMILES-T5 drug features, BioT5+ target protein features, NCL undersampling and an XGBoost-LightGBM weighted ensemble classifier.

## Main functions

- Extract drug features from SMILES sequences using SMILES-T5.
- Extract target protein features from amino acid sequences using BioT5+.
- Concatenate drug and target features into 1536-dimensional DTI features.
- Apply NCL undersampling to reduce class imbalance in the training set.
- Run 10-fold cross-validation on the NCL-cleaned training set.
- Train on the training set and evaluate on the independent test set.
- Search ensemble weights and decision threshold by MCC.
- Provide a Flask web predictor for single and batch DTI prediction.

## Repository structure

```text
.
|-- configs/
|   `-- ensemble_config_300_search_mcc.json
|-- data/
|   |-- README.md
|   |-- train.csv
|   `-- test.csv
|-- examples/
|   `-- batch_example.csv
|-- models/
|   `-- README.md
|-- outputs/
|   `-- README.md
|-- src/
|   |-- apply_ncl_undersampling.py
|   |-- concat_smiles_biot5_features.py
|   |-- extract_biot5_features.py
|   |-- extract_smiles_t5_features.py
|   |-- run_10fold_cv.py
|   |-- run_independent_test.py
|   `-- train_xgb_lgb_ensemble.py
|-- web_predictor/
|   |-- app.py
|   |-- start_web.bat
|   |-- static/
|   `-- templates/
|-- requirements.txt
`-- README.md
```

## Data included

The repository includes the raw BindingDB-derived DTI split used by this project:

- `data/train.csv`: training set
- `data/test.csv`: independent test set

The CSV files should contain at least the following columns:

```text
SMILES,Target Sequence,Label
```

Feature tables are not included because they are much larger and can be regenerated from the raw CSV files and pretrained encoders.

## Models used in this project

The feature extraction step uses the following pretrained sequence encoders:

- Drug encoder: `hothousetx/smiles_t5`
  - Local directory: `smilesT5/`
  - Architecture in this project: T5Model
  - Hidden size: 768
- Target encoder: `QizhiPei/biot5-plus-base`
  - Local directory: `bioT5+/`
  - Architecture in this project: T5ForConditionalGeneration encoder
  - Hidden size: 768

Download the complete model and tokenizer files and place them under the repository root:

```text
smilesT5/
bioT5+/
```

The Flask web predictor also expects the trained tree models in the repository root:

```text
best_model_xgb.pkl
best_model_lgb.pkl
configs/ensemble_config_300_search_mcc.json
```

The pretrained model folders and trained model files are not committed because they are large.

## Installation

```bash
pip install -r requirements.txt
```

For GPU acceleration during feature extraction, install a PyTorch version that matches your CUDA environment. The XGBoost and LightGBM scripts in this repository use CPU histogram/tree training by default for reproducibility and lower memory pressure.

## Step 1: feature extraction

Extract SMILES-T5 features for the training and independent test sets:

```bash
python src/extract_smiles_t5_features.py --input data/train.csv --model_dir smilesT5 --output data/train_smilesT5.csv
python src/extract_smiles_t5_features.py --input data/test.csv --model_dir smilesT5 --output data/test_smilesT5.csv
```

Extract BioT5+ features:

```bash
python src/extract_biot5_features.py --input data/train.csv --model_dir bioT5+ --output data/train_bioT5.csv
python src/extract_biot5_features.py --input data/test.csv --model_dir bioT5+ --output data/test_bioT5.csv
```

Concatenate drug and target features:

```bash
python src/concat_smiles_biot5_features.py --original data/train.csv --smiles_features data/train_smilesT5.csv --target_features data/train_bioT5.csv --output data/train_smilesT5_bioT5.csv
python src/concat_smiles_biot5_features.py --original data/test.csv --smiles_features data/test_smilesT5.csv --target_features data/test_bioT5.csv --output data/test_smilesT5_bioT5.csv
```

## Step 2: NCL undersampling

Apply NCL only to the training feature table. Do not resample the independent test set.

```bash
python src/apply_ncl_undersampling.py --input data/train_smilesT5_bioT5.csv --output data/ncl_train_smilesT5_bioT5.csv
```

## Step 3: 10-fold cross-validation

Run 10-fold cross-validation on the NCL-cleaned training set:

```bash
python src/run_10fold_cv.py
```

Default input and output paths:

- Input: `data/ncl_train_smilesT5_bioT5.csv`
- Output: `outputs/10fold_cv/`

This script does not use the independent test set. It only evaluates the model within the training set by stratified 10-fold cross-validation.

## Step 4: independent test evaluation

Train on the NCL-cleaned training set and evaluate on the independent test set:

```bash
python src/run_independent_test.py
```

Default input and output paths:

- Training data: `data/ncl_train_smilesT5_bioT5.csv`
- Independent test data: `data/test_smilesT5_bioT5.csv`
- Output: `outputs/independent_test/`

The script searches XGBoost-LightGBM ensemble weight and decision threshold on the evaluation labels by maximizing MCC, and saves the trained models and prediction results.

## Default hyperparameters

Default hyperparameters are consistent with the final paper setting:

- XGBoost `n_estimators = 6000`
- XGBoost `max_depth = 8`
- XGBoost `learning_rate = 0.02`
- XGBoost `subsample = 0.8`
- XGBoost `colsample_bytree = 0.8`
- LightGBM `n_estimators = 5000`
- LightGBM `learning_rate = 0.03`
- LightGBM `num_leaves = 95`
- LightGBM `feature_fraction = 0.8`
- Random seed `42`

The final ensemble setting used in the paper is stored in `configs/ensemble_config_300_search_mcc.json`:

- XGBoost weight = `0.56`
- LightGBM weight = `0.44`
- Decision threshold = `0.449`

Most paths and model hyperparameters can be changed from the command line. For example:

```bash
python src/run_10fold_cv.py --data data/ncl_train_smilesT5_bioT5.csv --n_estimators_xgb 6000 --n_estimators_lgb 5000 --xgb_max_depth 8
python src/run_independent_test.py --train data/ncl_train_smilesT5_bioT5.csv --test data/test_smilesT5_bioT5.csv --weight_step 0.01 --threshold_step 0.001
```

## Web predictor

Before starting the web predictor, make sure the required model files are placed in the repository root:

```text
smilesT5/
bioT5+/
best_model_xgb.pkl
best_model_lgb.pkl
configs/ensemble_config_300_search_mcc.json
```

Start the Flask web predictor:

```bash
cd web_predictor
python app.py
```

On Windows, you can also double-click or run:

```bash
web_predictor/start_web.bat
```

`start_web.bat` is only a convenience launcher. It changes the working directory to `web_predictor`, runs `python app.py`, and keeps the command window open after execution.

Then open:

```text
http://127.0.0.1:5000
```

If the required model files are missing, the web page will show a missing-file warning. Place the missing files in the repository root and restart the Flask app.

The web predictor supports:

- Single prediction: input one SMILES and one protein sequence.
- Batch prediction: upload a CSV file and download predicted results.

Batch prediction accepts these column names:

- Drug column: `SMILES`, `smiles`, or `drug_smiles`
- Protein column: `Protein`, `Target Sequence`, `Sequence`, or `protein_sequence`

## Relationship to the original project scripts

The scripts in `src/` were organized from the local experimental code used in this project. The GitHub version keeps the same experimental logic but removes local absolute paths, converts Chinese comments to English, exposes key parameters through `argparse`, and separates 10-fold cross-validation from independent test evaluation.

## Citation

If this code is useful for your research, please cite the corresponding paper or thesis related to SMILES-BioDTIPre.

## Notes

This repository is intended for research use. Predictions from computational DTI models should be further validated by molecular docking, molecular dynamics simulation or biological experiments.

