from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norm, 1e-12, None)


def fit_class_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    num_prototypes: int,
    seed: int,
) -> list[np.ndarray]:
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    labels = np.asarray(labels)
    all_prototypes = []
    for class_index in range(num_classes):
        class_embeddings = embeddings[labels == class_index]
        if len(class_embeddings) == 0:
            raise ValueError(f"Class {class_index} has no training samples")
        k = min(int(num_prototypes), len(class_embeddings))
        if k == 1:
            centers = class_embeddings.mean(axis=0, keepdims=True)
        else:
            centers = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(class_embeddings).cluster_centers_
        all_prototypes.append(l2_normalize(centers.astype(np.float32)))
    return all_prototypes


def prototype_logits(embeddings: np.ndarray, prototypes: list[np.ndarray], temperature: float) -> np.ndarray:
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    scores = []
    for class_centers in prototypes:
        similarity = embeddings @ class_centers.T
        scores.append(similarity.max(axis=1))
    return np.stack(scores, axis=1) / float(temperature)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
