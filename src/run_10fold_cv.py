from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, auc, matthews_corrcoef, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SCRIPT_VERSION = "smiles_biodtipre_10fold_cv_v1"


def find_label_column(df: pd.DataFrame) -> str:
    for name in ["Label", "label", "Y", "y", "target", "Target"]:
        if name in df.columns:
            return name
    return df.columns[-1]


def load_xy(path: Path):
    data = pd.read_csv(path)
    label_col = find_label_column(data)
    y = data[label_col].astype(int).values
    x = data.drop(columns=[label_col]).select_dtypes(include=[np.number]).astype(np.float32).values
    return x, y


def calculate_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "SN": tp / (tp + fn) if (tp + fn) else 0.0,
        "SP": tn / (tn + fp) if (tn + fp) else 0.0,
        "ACC": accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "AUPR": auc(recall, precision),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def make_xgb(args):
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=args.n_estimators_xgb,
        learning_rate=args.xgb_learning_rate,
        max_depth=args.xgb_max_depth,
        min_child_weight=args.xgb_min_child_weight,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        gamma=args.xgb_gamma,
        reg_alpha=args.xgb_reg_alpha,
        reg_lambda=args.xgb_reg_lambda,
        eval_metric="logloss",
        tree_method=args.xgb_tree_method,
        max_bin=args.xgb_max_bin,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbosity=1,
    )


def make_lgb(args):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators_lgb,
        learning_rate=args.lgb_learning_rate,
        num_leaves=args.lgb_num_leaves,
        max_depth=args.lgb_max_depth,
        min_child_samples=args.lgb_min_child_samples,
        feature_fraction=args.lgb_feature_fraction,
        bagging_fraction=args.lgb_bagging_fraction,
        bagging_freq=args.lgb_bagging_freq,
        lambda_l1=args.lgb_lambda_l1,
        lambda_l2=args.lgb_lambda_l2,
        max_bin=args.lgb_max_bin,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 10-fold cross-validation for the XGBoost-LightGBM ensemble.")
    parser.add_argument("--data", default=Path("data/ncl_train_smilesT5_bioT5.csv"), type=Path)
    parser.add_argument("--output_dir", default=Path("outputs/10fold_cv"), type=Path)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--xgb_weight", type=float, default=0.56)
    parser.add_argument("--lgb_weight", type=float, default=0.44)
    parser.add_argument("--threshold", type=float, default=0.449)
    parser.add_argument("--n_estimators_xgb", type=int, default=6000)
    parser.add_argument("--n_estimators_lgb", type=int, default=5000)
    parser.add_argument("--xgb_max_depth", type=int, default=8)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.02)
    parser.add_argument("--xgb_min_child_weight", type=float, default=0.1)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_gamma", type=float, default=1e-5)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.05)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    parser.add_argument("--xgb_tree_method", default="hist")
    parser.add_argument("--xgb_max_bin", type=int, default=256)
    parser.add_argument("--lgb_learning_rate", type=float, default=0.03)
    parser.add_argument("--lgb_num_leaves", type=int, default=95)
    parser.add_argument("--lgb_max_depth", type=int, default=-1)
    parser.add_argument("--lgb_min_child_samples", type=int, default=10)
    parser.add_argument("--lgb_feature_fraction", type=float, default=0.8)
    parser.add_argument("--lgb_bagging_fraction", type=float, default=1.0)
    parser.add_argument("--lgb_bagging_freq", type=int, default=1)
    parser.add_argument("--lgb_lambda_l1", type=float, default=0.0)
    parser.add_argument("--lgb_lambda_l2", type=float, default=0.0)
    parser.add_argument("--lgb_max_bin", type=int, default=255)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Script version:", SCRIPT_VERSION)
    print("Data:", args.data)
    print("Output dir:", args.output_dir)
    print("Ensemble weight:", {"xgb": args.xgb_weight, "lgb": args.lgb_weight})
    print("Threshold:", args.threshold)

    x, y = load_xy(args.data)
    print("X shape:", x.shape)
    print("Label counts:", pd.Series(y).value_counts().sort_index().to_dict())

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    fold_rows = []
    oof_rows = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y), start=1):
        print(f"\n[Ensemble] Fold {fold}/{args.n_splits}")
        x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model_xgb = make_xgb(args)
        model_lgb = make_lgb(args)

        print("Training XGBoost...")
        model_xgb.fit(x_train, y_train)
        print("Training LightGBM...")
        model_lgb.fit(x_train, y_train)

        prob_xgb = model_xgb.predict_proba(x_valid)[:, 1]
        prob_lgb = model_lgb.predict_proba(x_valid)[:, 1]
        prob_ensemble = args.xgb_weight * prob_xgb + args.lgb_weight * prob_lgb
        pred_ensemble = (prob_ensemble >= args.threshold).astype(int)
        metrics = calculate_metrics(y_valid, prob_ensemble, args.threshold)

        row = {"Fold": fold}
        row.update(metrics)
        fold_rows.append(row)

        for idx, yt, px, pl, pe, yp in zip(valid_idx, y_valid, prob_xgb, prob_lgb, prob_ensemble, pred_ensemble):
            oof_rows.append({
                "row_index": int(idx),
                "Fold": fold,
                "y_true": int(yt),
                "prob_xgb": float(px),
                "prob_lgb": float(pl),
                "prob_ensemble": float(pe),
                "pred_ensemble": int(yp),
                "threshold": float(args.threshold),
            })

        pd.DataFrame(fold_rows).to_csv(args.output_dir / "10fold_fold_details.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(oof_rows).to_csv(args.output_dir / "10fold_oof_predictions.csv", index=False, encoding="utf-8-sig")
        print(
            f"Fold {fold}: SN={metrics['SN']:.4f}, SP={metrics['SP']:.4f}, "
            f"ACC={metrics['ACC']:.4f}, MCC={metrics['MCC']:.4f}, "
            f"AUC={metrics['AUC']:.4f}, AUPR={metrics['AUPR']:.4f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "Model": "XGBoost_LightGBM_Ensemble",
        "CV": f"{args.n_splits}-fold",
        "XGB_weight": args.xgb_weight,
        "LGB_weight": args.lgb_weight,
        "Threshold": args.threshold,
    }
    for col in ["SN", "SP", "ACC", "MCC", "AUC", "AUPR", "TP", "FP", "FN", "TN"]:
        summary[col] = fold_df[col].mean()
    pd.DataFrame([summary]).to_csv(args.output_dir / "results_10fold_cv.csv", index=False, encoding="utf-8-sig")

    print("\n===== 10-fold cross-validation summary =====")
    for col in ["SN", "SP", "ACC", "MCC", "AUC", "AUPR"]:
        print(f"{col}: {summary[col]:.6f}")
    print("Saved outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
