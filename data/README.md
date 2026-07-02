# Data directory

This directory contains the raw BindingDB-derived DTI split used in this project:

```text
train.csv    training set
test.csv     independent test set
```

Recommended input columns:

```text
SMILES,Target Sequence,Label
```

Generated feature tables are intentionally ignored by `.gitignore` because they can be much larger than the raw dataset:

```text
train_smilesT5.csv
test_smilesT5.csv
train_bioT5.csv
test_bioT5.csv
train_smilesT5_bioT5.csv
test_smilesT5_bioT5.csv
ncl_train_smilesT5_bioT5.csv
```
