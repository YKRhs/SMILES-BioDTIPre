from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import NeighbourhoodCleaningRule


def find_label_column(df: pd.DataFrame) -> str:
    for name in ["Label", "label", "Y", "y", "target", "Target"]:
        if name in df.columns:
            return name
    return df.columns[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply NCL undersampling to the training feature table.")
    parser.add_argument("--input", default=Path("data/train_smilesT5_bioT5.csv"), type=Path, help="Input training feature CSV.")
    parser.add_argument("--output", default=Path("data/ncl_train_smilesT5_bioT5.csv"), type=Path, help="Output NCL-cleaned training CSV.")
    parser.add_argument("--label_col", default=None, help="Label column name. If omitted, it is inferred automatically.")
    parser.add_argument("--sampling_strategy", default="auto", help="NCL sampling strategy.")
    parser.add_argument("--n_neighbors", type=int, default=3, help="Number of nearest neighbors used by NCL.")
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    label_col = args.label_col or find_label_column(df)
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    y = df[label_col].astype(int).values
    x_df = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).astype(np.float32)

    print("Input:", args.input)
    print("X shape before NCL:", x_df.shape)
    print("Label counts before NCL:", pd.Series(y).value_counts().sort_index().to_dict())

    ncl = NeighbourhoodCleaningRule(
        sampling_strategy=args.sampling_strategy,
        n_neighbors=args.n_neighbors,
        n_jobs=args.n_jobs,
    )
    x_resampled, y_resampled = ncl.fit_resample(x_df.values, y)

    out_df = pd.DataFrame(x_resampled, columns=x_df.columns)
    out_df[label_col] = y_resampled.astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("X shape after NCL:", out_df.drop(columns=[label_col]).shape)
    print("Label counts after NCL:", pd.Series(y_resampled).value_counts().sort_index().to_dict())
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

