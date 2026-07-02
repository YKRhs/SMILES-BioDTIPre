from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


def clean_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper()


def add_space_to_sequence(sequence: str) -> str:
    return " ".join(clean_sequence(sequence))


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


@torch.no_grad()
def encode_sequences(sequences, tokenizer, model, device, max_length: int, batch_size: int) -> np.ndarray:
    features = []
    spaced_sequences = [add_space_to_sequence(seq) for seq in sequences]
    for start in tqdm(range(0, len(spaced_sequences), batch_size), desc="BioT5+"):
        batch = spaced_sequences[start:start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        features.append(pooled.detach().float().cpu().numpy())
    return np.concatenate(features, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BioT5+ target protein features.")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV with a target protein sequence column.")
    parser.add_argument("--model_dir", default="bioT5+", help="Local BioT5+ model directory.")
    parser.add_argument("--output", default="data/train_bioT5.csv", help="Output feature CSV.")
    parser.add_argument("--sequence_col", default="Target Sequence", help="Protein sequence column name.")
    parser.add_argument("--smiles_col", default="SMILES", help="Optional SMILES column to keep.")
    parser.add_argument("--label_col", default="Label", help="Optional label column to keep.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    df = pd.read_csv(args.input)
    if args.sequence_col not in df.columns:
        raise ValueError(f"Column not found: {args.sequence_col}")

    device = torch.device(args.device)
    dtype_kwargs = {"dtype": torch.float16} if device.type == "cuda" else {}

    tokenizer = T5Tokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(str(model_dir), local_files_only=True, **dtype_kwargs).to(device)
    model.eval()

    sequences = df[args.sequence_col].astype(str).tolist()
    embeddings = encode_sequences(sequences, tokenizer, model, device, args.max_length, args.batch_size)

    out = pd.DataFrame(embeddings, columns=[f"target_{i}" for i in range(embeddings.shape[1])])
    keep_cols = [col for col in [args.smiles_col, args.label_col] if col in df.columns]
    result = pd.concat([df[keep_cols].copy(), out], axis=1)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}, shape={result.shape}")


if __name__ == "__main__":
    main()

