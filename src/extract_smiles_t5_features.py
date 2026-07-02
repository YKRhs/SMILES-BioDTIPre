from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


@torch.no_grad()
def encode_smiles(smiles_list, tokenizer, model, device, max_length: int, batch_size: int) -> np.ndarray:
    features = []
    for start in tqdm(range(0, len(smiles_list), batch_size), desc="SMILES-T5"):
        batch = smiles_list[start:start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.encoder(**inputs)
        pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        features.append(pooled.detach().float().cpu().numpy())
    return np.concatenate(features, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SMILES-T5 drug features.")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV with a SMILES column.")
    parser.add_argument("--model_dir", default="smilesT5", help="Local SMILES-T5 model directory.")
    parser.add_argument("--output", default="data/train_smilesT5.csv", help="Output feature CSV.")
    parser.add_argument("--smiles_col", default="SMILES", help="SMILES column name.")
    parser.add_argument("--id_col", default=None, help="Optional ID column to keep.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    df = pd.read_csv(input_path)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column not found: {args.smiles_col}")

    device = torch.device(args.device)
    dtype_kwargs = {"dtype": torch.float16} if device.type == "cuda" else {}

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_dir / "tokenizer.json"),
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), local_files_only=True, **dtype_kwargs).to(device)
    model.eval()

    smiles_list = df[args.smiles_col].astype(str).tolist()
    embeddings = encode_smiles(smiles_list, tokenizer, model, device, args.max_length, args.batch_size)

    out = pd.DataFrame(embeddings, columns=[f"drug_{i}" for i in range(embeddings.shape[1])])
    keep_cols = []
    if args.id_col and args.id_col in df.columns:
        keep_cols.append(args.id_col)
    keep = df[[args.smiles_col] + keep_cols].copy()
    result = pd.concat([keep, out], axis=1)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}, shape={result.shape}")


if __name__ == "__main__":
    main()

