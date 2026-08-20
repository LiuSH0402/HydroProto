from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler


LABELS = [
    "EC3.1_ester_bonds",
    "EC3.2_glycosylases",
    "EC3.4_peptidases",
    "EC3.5_CN_nonpeptide",
    "EC3_other",
]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sequence_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    required = {"accession", "label", "sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame["accession"].isna().any() or frame["sequence"].isna().any():
        raise ValueError(f"{path} contains an empty accession or sequence")
    if frame["accession"].duplicated().any():
        raise ValueError(f"{path} contains duplicate accessions")
    unknown_labels = sorted(set(frame["label"]) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"{path} contains unknown labels: {unknown_labels}")
    normalized = frame["sequence"].astype(str).str.upper().str.replace(" ", "", regex=False)
    invalid = normalized.map(lambda seq: not set(seq).issubset(ALLOWED_AA))
    if invalid.any():
        bad = frame.loc[invalid, "accession"].head(5).tolist()
        raise ValueError(f"{path} contains invalid amino-acid symbols; example accessions: {bad}")
    frame = frame.copy()
    frame["sequence"] = normalized
    return frame


def encode_labels(labels: pd.Series | np.ndarray | list[str]) -> np.ndarray:
    return np.asarray([LABEL_TO_INDEX[str(label)] for label in labels], dtype=np.int64)


def make_dev_split(labels: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return np.sort(train_idx), np.sort(val_idx)


@dataclass(frozen=True)
class EmbeddingBundle:
    accessions: np.ndarray
    labels: np.ndarray
    embeddings: np.ndarray
    lengths: np.ndarray
    metadata: dict


def load_embedding_bundle(path: str | Path) -> EmbeddingBundle:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"accessions", "labels", "embeddings", "lengths", "metadata_json"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing embedding fields: {sorted(missing)}")
        import json

        metadata = json.loads(str(archive["metadata_json"].item()))
        bundle = EmbeddingBundle(
            accessions=archive["accessions"].astype(str),
            labels=archive["labels"].astype(str),
            embeddings=archive["embeddings"].astype(np.float32),
            lengths=archive["lengths"].astype(np.int32),
            metadata=metadata,
        )
    n = len(bundle.accessions)
    if not (len(bundle.labels) == len(bundle.embeddings) == len(bundle.lengths) == n):
        raise ValueError(f"{path} contains embedding arrays with inconsistent lengths")
    if not np.isfinite(bundle.embeddings).all():
        raise ValueError(f"{path} contains NaN or Inf values")
    return bundle


def verify_bundle_matches_csv(bundle: EmbeddingBundle, frame: pd.DataFrame, source_path: str | Path) -> None:
    csv_ids = frame["accession"].astype(str).to_numpy()
    csv_labels = frame["label"].astype(str).to_numpy()
    if not np.array_equal(bundle.accessions, csv_ids):
        raise ValueError("Embedding accession order does not match the CSV")
    if not np.array_equal(bundle.labels, csv_labels):
        raise ValueError("Embedding labels do not match the CSV")
    expected_hash = sha256_file(source_path)
    stored_hash = bundle.metadata.get("input_sha256")
    if stored_hash != expected_hash:
        raise ValueError(f"Embedding input hash mismatch: {stored_hash} != {expected_hash}")


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, return_index: bool = False):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.return_index = bool(return_index)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        if self.return_index:
            return self.embeddings[index], self.labels[index], int(index)
        return self.embeddings[index], self.labels[index]


class BalancedBatchSampler(Sampler[list[int]]):
    """Sample the same number of records from every class in each batch."""

    def __init__(self, labels: np.ndarray, samples_per_class: int, seed: int):
        self.labels = np.asarray(labels)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.classes = np.unique(self.labels)
        self.by_class = {cls: np.flatnonzero(self.labels == cls) for cls in self.classes}
        if self.samples_per_class < 2:
            raise ValueError("Supervised contrastive learning requires at least two samples per class per batch")
        if any(len(indices) == 0 for indices in self.by_class.values()):
            raise ValueError("At least one class has no samples")
        self.batch_size = len(self.classes) * self.samples_per_class
        self.num_batches = int(np.ceil(len(self.labels) / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {}
        pointers = {}
        for cls, indices in self.by_class.items():
            pools[cls] = rng.permutation(indices)
            pointers[cls] = 0
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                pool = pools[cls]
                pointer = pointers[cls]
                need = self.samples_per_class
                selected = []
                while need > 0:
                    remaining = len(pool) - pointer
                    take = min(need, remaining)
                    selected.extend(pool[pointer:pointer + take].tolist())
                    pointer += take
                    need -= take
                    if pointer == len(pool):
                        pool = rng.permutation(self.by_class[cls])
                        pointer = 0
                pools[cls] = pool
                pointers[cls] = pointer
                batch.extend(selected)
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def build_local_positive_graph(
    embeddings: np.ndarray,
    labels: np.ndarray,
    neighbors_per_sample: int = 10,
    mutual: bool = True,
) -> tuple[np.ndarray, dict]:
    """Build a same-class local-neighbor graph without using validation records."""

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    normalized = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    graph = np.zeros((len(labels), len(labels)), dtype=bool)
    directed = np.zeros_like(graph)
    for class_index in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_index)
        class_vectors = normalized[class_indices]
        similarities = class_vectors @ class_vectors.T
        np.fill_diagonal(similarities, -np.inf)
        k = min(int(neighbors_per_sample), len(class_indices) - 1)
        if k < 1:
            raise ValueError(f"Class {class_index} needs at least two training records")
        nearest_local = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
        rows = np.repeat(class_indices, k)
        columns = class_indices[nearest_local.reshape(-1)]
        directed[rows, columns] = True

    graph[:] = directed & directed.T if mutual else directed | directed.T
    graph |= graph.T
    np.fill_diagonal(graph, False)

    # Mutual kNN can leave a small number of isolated records. Add their nearest
    # same-class neighbor so every anchor can participate in local contrastive loss.
    fallback_edges = 0
    for index in np.flatnonzero(graph.sum(axis=1) == 0):
        same_class = np.flatnonzero(labels == labels[index])
        same_class = same_class[same_class != index]
        similarities = normalized[same_class] @ normalized[index]
        neighbor = int(same_class[int(np.argmax(similarities))])
        graph[index, neighbor] = True
        graph[neighbor, index] = True
        fallback_edges += 1

    degrees = graph.sum(axis=1)
    edge_count = int(np.triu(graph, k=1).sum())
    stats = {
        "records": int(len(labels)),
        "neighbors_per_sample": int(neighbors_per_sample),
        "mutual_knn": bool(mutual),
        "undirected_edges": edge_count,
        "fallback_edges": int(fallback_edges),
        "degree_min": int(degrees.min()),
        "degree_mean": float(degrees.mean()),
        "degree_median": float(np.median(degrees)),
        "degree_max": int(degrees.max()),
    }
    return graph, stats


class LocalPositiveBatchSampler(Sampler[list[int]]):
    """Balanced batches containing explicit same-class local-neighbor pairs."""

    def __init__(self, labels: np.ndarray, positive_graph: np.ndarray, samples_per_class: int, seed: int):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.positive_graph = np.asarray(positive_graph, dtype=bool)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.classes = np.unique(self.labels)
        self.by_class = {cls: np.flatnonzero(self.labels == cls) for cls in self.classes}
        if self.positive_graph.shape != (len(self.labels), len(self.labels)):
            raise ValueError("positive_graph shape does not match labels")
        if self.samples_per_class < 2 or self.samples_per_class % 2:
            raise ValueError("Local-positive batches require an even samples_per_class >= 2")
        if np.any(self.positive_graph.sum(axis=1) == 0):
            raise ValueError("Every record must have at least one local positive")
        self.batch_size = len(self.classes) * self.samples_per_class
        self.num_batches = int(np.ceil(len(self.labels) / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pairs_per_class = self.samples_per_class // 2
        anchor_pools = {cls: rng.permutation(indices) for cls, indices in self.by_class.items()}
        pointers = {cls: 0 for cls in self.classes}
        for _ in range(self.num_batches):
            batch: list[int] = []
            for cls in self.classes:
                pool = anchor_pools[cls]
                pointer = pointers[cls]
                if pointer + pairs_per_class > len(pool):
                    pool = rng.permutation(self.by_class[cls])
                    pointer = 0
                anchors = pool[pointer:pointer + pairs_per_class]
                pointer += pairs_per_class
                anchor_pools[cls] = pool
                pointers[cls] = pointer
                for anchor in anchors:
                    candidates = np.flatnonzero(self.positive_graph[anchor])
                    positive = int(rng.choice(candidates))
                    batch.extend([int(anchor), positive])
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
