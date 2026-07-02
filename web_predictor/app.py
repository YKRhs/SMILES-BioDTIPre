from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast, T5ForConditionalGeneration, T5Tokenizer


APP_VERSION = "SMILES-BioDTIPre web predictor v1"

WEB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEB_DIR.parent
OUTPUT_DIR = WEB_DIR / "outputs"

SMILES_MODEL_DIR = PROJECT_DIR / "smilesT5"
TARGET_MODEL_DIR = PROJECT_DIR / "bioT5+"
XGB_MODEL_PATH = PROJECT_DIR / "best_model_xgb.pkl"
LGB_MODEL_PATH = PROJECT_DIR / "best_model_lgb.pkl"
CONFIG_PATH = PROJECT_DIR / "ensemble_config_300_search_mcc.json"
CONFIG_PATH_IN_CONFIGS = PROJECT_DIR / "configs" / "ensemble_config_300_search_mcc.json"

MAX_LENGTH = 512
DEFAULT_XGB_WEIGHT = 0.56
DEFAULT_LGB_WEIGHT = 0.44
DEFAULT_THRESHOLD = 0.449

app = Flask(__name__)
OUTPUT_DIR.mkdir(exist_ok=True)

_state_lock = threading.Lock()
_model_state = {
    "loaded": False,
    "loading": False,
    "device": None,
    "smiles_tokenizer": None,
    "smiles_model": None,
    "target_tokenizer": None,
    "target_model": None,
    "xgb_model": None,
    "lgb_model": None,
    "config": None,
    "loaded_at": None,
}


def load_config() -> dict:
    config = {
        "best_xgb_weight": DEFAULT_XGB_WEIGHT,
        "best_lgb_weight": DEFAULT_LGB_WEIGHT,
        "best_threshold": DEFAULT_THRESHOLD,
        "source": "default",
    }
    config_path = CONFIG_PATH if CONFIG_PATH.exists() else CONFIG_PATH_IN_CONFIGS
    if config_path.exists():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(loaded)
        config["source"] = str(config_path)
    return config


def require_files() -> list[str]:
    missing = []
    for path in [SMILES_MODEL_DIR, TARGET_MODEL_DIR, XGB_MODEL_PATH, LGB_MODEL_PATH]:
        if not path.exists():
            missing.append(str(path))
    return missing


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all_models() -> None:
    with _state_lock:
        if _model_state["loaded"]:
            return
        if _model_state["loading"]:
            raise RuntimeError("Models are already loading. Please wait and try again.")
        _model_state["loading"] = True

    try:
        missing = require_files()
        if missing:
            raise FileNotFoundError("Missing required model files: " + "; ".join(missing))

        device = get_device()
        dtype_kwargs = {"dtype": torch.float16} if device.type == "cuda" else {}

        smiles_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(SMILES_MODEL_DIR / "tokenizer.json"),
            pad_token="<pad>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        smiles_model = AutoModelForSeq2SeqLM.from_pretrained(
            str(SMILES_MODEL_DIR),
            local_files_only=True,
            **dtype_kwargs,
        ).to(device)
        smiles_model.eval()

        target_tokenizer = T5Tokenizer.from_pretrained(str(TARGET_MODEL_DIR), local_files_only=True)
        target_model = T5ForConditionalGeneration.from_pretrained(
            str(TARGET_MODEL_DIR),
            local_files_only=True,
            **dtype_kwargs,
        ).to(device)
        target_model.eval()

        xgb_model = joblib.load(XGB_MODEL_PATH)
        lgb_model = joblib.load(LGB_MODEL_PATH)

        config = load_config()

        with _state_lock:
            _model_state.update(
                {
                    "loaded": True,
                    "loading": False,
                    "device": device,
                    "smiles_tokenizer": smiles_tokenizer,
                    "smiles_model": smiles_model,
                    "target_tokenizer": target_tokenizer,
                    "target_model": target_model,
                    "xgb_model": xgb_model,
                    "lgb_model": lgb_model,
                    "config": config,
                    "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    except Exception:
        with _state_lock:
            _model_state["loading"] = False
        raise


def clean_protein_sequence(sequence: str) -> str:
    sequence = re.sub(r"\s+", "", sequence or "").upper()
    return sequence


def add_space_to_sequence(sequence: str) -> str:
    return " ".join(clean_protein_sequence(sequence))


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


@torch.no_grad()
def extract_smiles_feature(smiles: str) -> np.ndarray:
    tokenizer = _model_state["smiles_tokenizer"]
    model = _model_state["smiles_model"]
    device = _model_state["device"]

    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.encoder(**inputs)
    feature = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return feature.squeeze(0).detach().float().cpu().numpy()


@torch.no_grad()
def extract_target_feature(sequence: str) -> np.ndarray:
    tokenizer = _model_state["target_tokenizer"]
    model = _model_state["target_model"]
    device = _model_state["device"]

    spaced_sequence = add_space_to_sequence(sequence)
    inputs = tokenizer(
        spaced_sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    feature = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return feature.squeeze(0).detach().float().cpu().numpy()


def predict_one(smiles: str, sequence: str) -> dict:
    smiles = (smiles or "").strip()
    sequence = clean_protein_sequence(sequence)
    if not smiles:
        raise ValueError("Please enter a SMILES string.")
    if not sequence:
        raise ValueError("Please enter a target protein sequence.")

    load_all_models()

    drug_feature = extract_smiles_feature(smiles)
    target_feature = extract_target_feature(sequence)
    x = np.concatenate([drug_feature, target_feature]).astype(np.float32).reshape(1, -1)

    if x.shape[1] != 1536:
        raise RuntimeError(f"Feature dimension mismatch: expected 1536, got {x.shape[1]}.")

    xgb_model = _model_state["xgb_model"]
    lgb_model = _model_state["lgb_model"]
    config = _model_state["config"]

    prob_xgb = float(xgb_model.predict_proba(x)[0, 1])
    prob_lgb = float(lgb_model.predict_proba(x)[0, 1])
    w_xgb = float(config.get("best_xgb_weight", DEFAULT_XGB_WEIGHT))
    w_lgb = float(config.get("best_lgb_weight", DEFAULT_LGB_WEIGHT))
    threshold = float(config.get("best_threshold", DEFAULT_THRESHOLD))
    prob = w_xgb * prob_xgb + w_lgb * prob_lgb
    label = int(prob >= threshold)

    return {
        "smiles": smiles,
        "sequence_length": len(sequence),
        "prob_xgb": prob_xgb,
        "prob_lgb": prob_lgb,
        "probability": float(prob),
        "threshold": threshold,
        "xgb_weight": w_xgb,
        "lgb_weight": w_lgb,
        "prediction": label,
        "prediction_text": "Interaction" if label == 1 else "Non-interaction",
        "device": str(_model_state["device"]),
    }


def find_column(columns, candidates):
    normalized = {str(col).strip().lower(): col for col in columns}
    for name in candidates:
        key = name.lower()
        if key in normalized:
            return normalized[key]
    return None


def predict_batch(file_storage) -> dict:
    if not file_storage or not file_storage.filename:
        raise ValueError("Please upload a CSV file.")

    file_bytes = file_storage.read()
    last_error = None
    df = None
    for encoding in ["utf-8-sig", "utf-8", "gbk", "gb18030", "utf-16", "utf-16le", "utf-16be", "latin1"]:
        try:
            df = pd.read_csv(BytesIO(file_bytes), encoding=encoding, sep=None, engine="python")
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    if df is None:
        raise ValueError(f"Failed to read CSV encoding: {last_error}")
    smiles_col = find_column(df.columns, ["SMILES", "smiles", "Drug SMILES"])
    protein_col = find_column(
        df.columns,
        ["Protein", "protein", "Target Sequence", "target_sequence", "Sequence", "sequence"],
    )
    if smiles_col is None or protein_col is None:
        available = ", ".join(str(col) for col in df.columns)
        raise ValueError(f"The CSV file must contain SMILES and Protein columns. Detected columns: {available}")

    rows = []
    total = len(df)
    if total == 0:
        raise ValueError("The uploaded CSV file is empty.")

    load_all_models()

    for idx, row in df.iterrows():
        smiles = "" if pd.isna(row[smiles_col]) else str(row[smiles_col])
        protein = "" if pd.isna(row[protein_col]) else str(row[protein_col])
        try:
            result = predict_one(smiles, protein)
            rows.append(
                {
                    "row_index": idx,
                    "SMILES": result["smiles"],
                    "Protein": clean_protein_sequence(protein),
                    "protein_length": result["sequence_length"],
                    "xgb_probability": result["prob_xgb"],
                    "lgb_probability": result["prob_lgb"],
                    "ensemble_probability": result["probability"],
                    "threshold": result["threshold"],
                    "prediction": result["prediction"],
                    "prediction_label": result["prediction_text"],
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "row_index": idx,
                    "SMILES": smiles,
                    "Protein": clean_protein_sequence(protein),
                    "protein_length": len(clean_protein_sequence(protein)),
                    "xgb_probability": np.nan,
                    "lgb_probability": np.nan,
                    "ensemble_probability": np.nan,
                    "threshold": np.nan,
                    "prediction": np.nan,
                    "prediction_label": "",
                    "error": str(exc),
                }
            )

    out_df = pd.DataFrame(rows)
    filename = f"smiles_biodtipre_batch_{uuid.uuid4().hex[:10]}.csv"
    out_path = OUTPUT_DIR / filename
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    ok_count = int((out_df["error"] == "").sum())
    positive_count = int((out_df["prediction"] == 1).sum())
    return {
        "total": total,
        "ok_count": ok_count,
        "error_count": total - ok_count,
        "positive_count": positive_count,
        "negative_count": ok_count - positive_count,
        "filename": filename,
        "preview": out_df.head(10).to_dict(orient="records"),
    }


@app.get("/")
def index():
    config = load_config()
    return render_template(
        "index.html",
        app_version=APP_VERSION,
        config=config,
        missing=require_files(),
    )


@app.get("/home")
@app.get("/predictor")
@app.get("/about")
def alias_pages():
    config = load_config()
    return render_template(
        "index.html",
        app_version=APP_VERSION,
        config=config,
        missing=require_files(),
    )


@app.get("/status")
def status():
    with _state_lock:
        payload = {
            "loaded": _model_state["loaded"],
            "loading": _model_state["loading"],
            "device": str(_model_state["device"]) if _model_state["device"] else None,
            "loaded_at": _model_state["loaded_at"],
            "missing": require_files(),
        }
    return jsonify(payload)


@app.post("/predict")
def predict():
    started = time.time()
    smiles = request.form.get("smiles", "")
    sequence = request.form.get("sequence", "")
    try:
        result = predict_one(smiles, sequence)
        result["elapsed_seconds"] = round(time.time() - started, 3)
        return render_template("result.html", result=result, error=None)
    except Exception as exc:
        return render_template(
            "result.html",
            result=None,
            error=str(exc),
            smiles=smiles,
            sequence=sequence,
        ), 400


@app.post("/batch_predict")
def batch_predict():
    started = time.time()
    try:
        payload = predict_batch(request.files.get("file"))
        payload["elapsed_seconds"] = round(time.time() - started, 3)
        return render_template("batch_result.html", result=payload, error=None)
    except Exception as exc:
        return render_template("batch_result.html", result=None, error=str(exc)), 400


@app.get("/download/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    host = os.environ.get("SMILES_BIODTI_HOST", "127.0.0.1")
    port = int(os.environ.get("SMILES_BIODTI_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
