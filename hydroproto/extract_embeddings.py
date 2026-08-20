from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import load_sequence_csv, sha256_file


MODEL_REGISTRY = {
    "esm2_t30_150M_UR50D": ("esm2_t30_150M_UR50D", 30, 640),
    "esm2_t12_35M_UR50D": ("esm2_t12_35M_UR50D", 12, 480),
}


@dataclass(frozen=True)
class ChunkRecord:
    sequence_index: int
    chunk_index: int
    sequence: str
    weight: int


def split_sequence(sequence: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if len(sequence) <= chunk_size:
        return [sequence]
    step = chunk_size - overlap
    starts = list(range(0, len(sequence) - chunk_size + 1, step))
    final_start = len(sequence) - chunk_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [sequence[start:start + chunk_size] for start in starts]


def build_chunks(sequences: list[str], chunk_size: int, overlap: int) -> list[ChunkRecord]:
    records = []
    for seq_index, sequence in enumerate(sequences):
        chunks = split_sequence(sequence, chunk_size, overlap)
        for chunk_index, chunk in enumerate(chunks):
            records.append(ChunkRecord(seq_index, chunk_index, chunk, len(chunk)))
    return records


def dynamic_batches(records: list[ChunkRecord], max_batch_tokens: int, max_batch_sequences: int):
    ordered = sorted(records, key=lambda rec: len(rec.sequence), reverse=True)
    batch = []
    tokens = 0
    for record in ordered:
        length = len(record.sequence) + 2
        if batch and (tokens + length > max_batch_tokens or len(batch) >= max_batch_sequences):
            yield batch
            batch = []
            tokens = 0
        batch.append(record)
        tokens += length
    if batch:
        yield batch


def load_esm_model(model_name: str, device: torch.device):
    try:
        import esm
    except ImportError as exc:
        raise RuntimeError("fair-esm is required; install fair-esm==2.0.0") from exc
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}; choices: {sorted(MODEL_REGISTRY)}")
    loader_name, layer, embedding_dim = MODEL_REGISTRY[model_name]
    loader = getattr(esm.pretrained, loader_name)
    model, alphabet = loader()
    model.eval().to(device)
    return model, alphabet, layer, embedding_dim


def extract(args) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = load_sequence_csv(input_path)
    if args.limit is not None:
        frame = frame.iloc[: args.limit].copy()
    sequences = frame["sequence"].tolist()
    records = build_chunks(sequences, args.chunk_size, args.overlap)
    batches = list(dynamic_batches(records, args.max_batch_tokens, args.max_batch_sequences))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, alphabet, layer, embedding_dim = load_esm_model(args.model, device)
    batch_converter = alphabet.get_batch_converter()
    sums = np.zeros((len(frame), embedding_dim), dtype=np.float64)
    weights = np.zeros(len(frame), dtype=np.float64)
    peak_memory = 0

    progress = tqdm(batches, desc="ESM-2 embedding batches", unit="batch")
    for batch_index, batch in enumerate(progress):
        batch_data = [
            (f"seq{record.sequence_index}_chunk{record.chunk_index}", record.sequence)
            for record in batch
        ]
        _, strings, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)
        amp_enabled = device.type == "cuda" and args.amp
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(tokens, repr_layers=[layer], return_contacts=False)
            representations = outputs["representations"][layer]
        for row_index, (record, sequence) in enumerate(zip(batch, strings)):
            pooled = representations[row_index, 1:len(sequence) + 1].float().mean(dim=0)
            vector = pooled.cpu().numpy().astype(np.float32)
            sums[record.sequence_index] += vector * record.weight
            weights[record.sequence_index] += record.weight
        del tokens, representations, outputs
        if device.type == "cuda":
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
        if batch_index % 25 == 0 and device.type == "cuda":
            progress.set_postfix(vram_gb=f"{peak_memory / 1024**3:.2f}")

    embeddings = (sums / weights[:, None]).astype(np.float32)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "records": int(len(frame)),
        "model": args.model,
        "representation_layer": int(layer),
        "embedding_dim": int(embedding_dim),
        "chunk_size": int(args.chunk_size),
        "overlap": int(args.overlap),
        "total_chunks": int(len(records)),
        "max_batch_tokens": int(args.max_batch_tokens),
        "max_batch_sequences": int(args.max_batch_sequences),
        "device": str(device),
        "amp": bool(args.amp),
        "peak_cuda_memory_bytes": int(peak_memory),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "limited_run": args.limit is not None,
    }
    np.savez_compressed(
        output_path,
        accessions=frame["accession"].astype(str).to_numpy(dtype=str),
        labels=frame["label"].astype(str).to_numpy(dtype=str),
        embeddings=embeddings,
        lengths=frame["sequence"].str.len().to_numpy(dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"saved={output_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract chunk-aware ESM-2 sequence embeddings")
    parser.add_argument("--input", required=True, help="Input CSV containing accession, label and sequence")
    parser.add_argument("--output", required=True, help="Output compressed NPZ")
    parser.add_argument("--model", default="esm2_t30_150M_UR50D", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--max-batch-tokens", type=int, default=2400)
    parser.add_argument("--max-batch-sequences", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)
    parser.add_argument("--limit", type=int, default=None, help="Only for smoke testing; use first N records")
    return parser.parse_args()


def main():
    extract(parse_args())


if __name__ == "__main__":
    main()
