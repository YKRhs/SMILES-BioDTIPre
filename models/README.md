# Model files

Large pretrained models and trained classifiers are not included in the repository.

Pretrained encoders used in this project:

```text
hothousetx/smiles_t5        -> ../smilesT5/
QizhiPei/biot5-plus-base    -> ../bioT5+/
```

Expected local files/directories for the web predictor:

```text
../smilesT5/
../bioT5+/
../best_model_xgb.pkl
../best_model_lgb.pkl
../configs/ensemble_config_300_search_mcc.json
```

The web predictor loads these files from the repository root.
