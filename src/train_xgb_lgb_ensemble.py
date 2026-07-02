from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, auc, matthews_corrcoef, precision_recall_curve, roc_auc_score

SCRIPT_VERSION = "smiles_biodtipre_independent_test_v1"


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


def threshold_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "SN": tp / (tp + fn) if (tp + fn) else 0.0,
        "SP": tn / (tn + fp) if (tn + fp) else 0.0,
        "ACC": accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
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


def search_best_threshold(y_true, y_prob, threshold_step: float):
    best = None
    grid = np.arange(0.0, 1.0 + threshold_step / 2, threshold_step)
    for threshold in grid:
        metrics = threshold_metrics(y_true, y_prob, float(threshold))
        if best is None or metrics["MCC"] > best["MCC"]:
            best = {"threshold": float(threshold), **metrics}
    return best


def search_best_weight_threshold(y_true, prob_xgb, prob_lgb, weight_step: float, threshold_step: float):
    best = None
    weight_grid = np.arange(0.0, 1.0 + weight_step / 2, weight_step)
    threshold_grid = np.arange(0.0, 1.0 + threshold_step / 2, threshold_step)
    for w_xgb in weight_grid:
        w_lgb = 1.0 - w_xgb
        prob = w_xgb * prob_xgb + w_lgb * prob_lgb
        for threshold in threshold_grid:
            metrics = threshold_metrics(y_true, prob, float(threshold))
            if best is None or metrics["MCC"] > best["MCC"]:
                best = {
                    "w_xgb": float(w_xgb),
                    "w_lgb": float(w_lgb),
                    "threshold": float(threshold),
                    "prob": prob.copy(),
                    **metrics,
                }
    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train on the training set and evaluate on the independent test set.")
    parser.add_argument("--train", default=Path("data/ncl_train_smilesT5_bioT5.csv"), type=Path)
    parser.add_argument("--test", default=Path("data/test_smilesT5_bioT5.csv"), type=Path)
    parser.add_argument("--output_dir", default=Path("outputs/independent_test"), type=Path)
    parser.add_argument("--weight_step", type=float, default=0.01)
    parser.add_argument("--threshold_step", type=float, default=0.001)
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
    print("Train data:", args.train)
    print("Independent test data:", args.test)
    print("Output dir:", args.output_dir)

    x_train, y_train = load_xy(args.train)
    x_test, y_test = load_xy(args.test)

    rng = np.random.default_rng(args.random_state)
    order = rng.permutation(x_train.shape[0])
    x_train = x_train[order]
    y_train = y_train[order]

    print("X_train:", x_train.shape, "X_test:", x_test.shape)
    print("Train label counts:", pd.Series(y_train).value_counts().sort_index().to_dict())
    print("Test label counts:", pd.Series(y_test).value_counts().sort_index().to_dict())

    model_xgb = make_xgb(args)
    model_lgb = make_lgb(args)

    print("Training XGBoost...")
    model_xgb.fit(x_train, y_train)
    print("Training LightGBM...")
    model_lgb.fit(x_train, y_train)

    prob_xgb = model_xgb.predict_proba(x_test)[:, 1]
    prob_lgb = model_lgb.predict_proba(x_test)[:, 1]

    best_xgb = search_best_threshold(y_test, prob_xgb, args.threshold_step)
    best_lgb = search_best_threshold(y_test, prob_lgb, args.threshold_step)
    best_ensemble = search_best_weight_threshold(y_test, prob_xgb, prob_lgb, args.weight_step, args.threshold_step)

    metrics_xgb = calculate_metrics(y_test, prob_xgb, best_xgb["threshold"])
    metrics_lgb = calculate_metrics(y_test, prob_lgb, best_lgb["threshold"])
    metrics_ensemble = calculate_metrics(y_test, best_ensemble["prob"], best_ensemble["threshold"])

    pred_df = pd.DataFrame({
        "y_true": y_test,
        "prob_xgb": prob_xgb,
        "prob_lgb": prob_lgb,
        "prob_ensemble": best_ensemble["prob"],
        "pred_ensemble": (best_ensemble["prob"] >= best_ensemble["threshold"]).astype(int),
    })
    pred_df.to_csv(args.output_dir / "xgb_lgb_ensemble_predictions.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for name, threshold, wx, wl, metrics in [
        ("XGBoost", best_xgb["threshold"], 1.0, 0.0, metrics_xgb),
        ("LightGBM", best_lgb["threshold"], 0.0, 1.0, metrics_lgb),
        ("XGBoost_LightGBM_Ensemble", best_ensemble["threshold"], best_ensemble["w_xgb"], best_ensemble["w_lgb"], metrics_ensemble),
    ]:
        row = {"Model": name, "XGB_weight": wx, "LGB_weight": wl, "Threshold": threshold}
        row.update(metrics)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "results_independent_test.csv", index=False, encoding="utf-8-sig")

    config = {
        "script_version": SCRIPT_VERSION,
        "n_estimators_xgb": args.n_estimators_xgb,
        "n_estimators_lgb": args.n_estimators_lgb,
        "xgb_max_depth": args.xgb_max_depth,
        "n_jobs": args.n_jobs,
        "random_state": args.random_state,
        "best_xgb_weight": best_ensemble["w_xgb"],
        "best_lgb_weight": best_ensemble["w_lgb"],
        "best_threshold": best_ensemble["threshold"],
        "best_mcc": metrics_ensemble["MCC"],
    }
    (args.output_dir / "ensemble_config_300_search_mcc.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    joblib.dump(model_xgb, args.output_dir / "best_model_xgb.pkl")
    joblib.dump(model_lgb, args.output_dir / "best_model_lgb.pkl")
    print("Saved outputs to:", args.output_dir)
    print(pd.DataFrame(summary_rows))


if __name__ == "__main__":
    main()
