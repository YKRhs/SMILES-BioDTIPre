from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...], exclude: set[str]) -> list[str]:
    cols = [col for col in df.columns if str(col).startswith(prefixes)]
    if cols:
        return cols
    return [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate SMILES-T5 and BioT5+ features.")
    parser.add_argument("--original", default="data/train.csv", help="Original CSV containing SMILES and Label.")
    parser.add_argument("--smiles_features", default="data/train_smilesT5.csv", help="SMILES-T5 feature CSV.")
    parser.add_argument("--target_features", default="data/train_bioT5.csv", help="BioT5+ feature CSV.")
    parser.add_argument("--output", default="data/train_smilesT5_bioT5.csv", help="Output concatenated feature CSV.")
    parser.add_argument("--smiles_col", default="SMILES")
    parser.add_argument("--label_col", default="Label")
    args = parser.parse_args()

    original_df = pd.read_csv(args.original)
    smiles_df = pd.read_csv(args.smiles_features)
    target_df = pd.read_csv(args.target_features)

    if len(original_df) != len(smiles_df) or len(original_df) != len(target_df):
        raise ValueError("Row counts do not match.")

    if args.smiles_col in original_df.columns and args.smiles_col in smiles_df.columns:
        if not original_df[args.smiles_col].astype(str).reset_index(drop=True).equals(
            smiles_df[args.smiles_col].astype(str).reset_index(drop=True)
        ):
            raise ValueError("SMILES-T5 feature file is not aligned with the original CSV.")

    if args.smiles_col in original_df.columns and args.smiles_col in target_df.columns:
        if not original_df[args.smiles_col].astype(str).reset_index(drop=True).equals(
            target_df[args.smiles_col].astype(str).reset_index(drop=True)
        ):
            raise ValueError("BioT5+ feature file is not aligned with the original CSV.")

    if args.label_col not in original_df.columns:
        raise ValueError(f"Label column not found in original CSV: {args.label_col}")

    drug_cols = feature_columns(smiles_df, ("drug_",), {args.smiles_col, "ID", args.label_col})
    target_cols = feature_columns(target_df, ("target_",), {args.smiles_col, "ID", args.label_col})

    drug = smiles_df[drug_cols].astype(np.float32).copy()
    target = target_df[target_cols].astype(np.float32).copy()
    drug.columns = [f"drug_{i}" for i in range(drug.shape[1])]
    target.columns = [f"target_{i}" for i in range(target.shape[1])]
    label = original_df[args.label_col].astype(int).rename(args.label_col)

    output_df = pd.concat([drug, target, label], axis=1)
    output_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}, shape={output_df.shape}")


if __name__ == "__main__":
    main()

