from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        logits = self.classifier(self.norm(x))
        return {"logits": logits, "projection": None, "features": x}


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.adapter = Adapter(input_dim, hidden_dim, feature_dim, dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.adapter(x)
        return {"logits": self.classifier(features), "projection": None, "features": features}


class MetricProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, projection_dim: int, dropout: float):
        super().__init__()
        self.adapter = Adapter(input_dim, hidden_dim, feature_dim, dropout)
        self.projector = nn.Linear(feature_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.adapter(x)
        projection = F.normalize(self.projector(features), dim=1)
        return {"logits": None, "projection": projection, "features": features}


class HydroMPC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        feature_dim: int,
        projection_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.adapter = Adapter(input_dim, hidden_dim, feature_dim, dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.projector = nn.Linear(feature_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.adapter(x)
        logits = self.classifier(features)
        projection = F.normalize(self.projector(features), dim=1)
        return {"logits": logits, "projection": projection, "features": features}


class HydroMPCV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        feature_dim: int,
        projection_dim: int,
        num_classes: int,
        num_prototypes: int,
        prototype_temperature: float,
        dropout: float,
        online_prototypes: bool,
    ):
        super().__init__()
        self.adapter = Adapter(input_dim, hidden_dim, feature_dim, dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.projector = nn.Linear(feature_dim, projection_dim)
        self.online_prototypes = bool(online_prototypes)
        self.num_classes = int(num_classes)
        self.num_prototypes = int(num_prototypes)
        self.prototype_temperature = float(prototype_temperature)
        if self.online_prototypes:
            initial = torch.randn(num_classes, num_prototypes, projection_dim)
            self.prototypes = nn.Parameter(F.normalize(initial, dim=2))
        else:
            self.register_parameter("prototypes", None)
        self.register_buffer("prototypes_initialized", torch.tensor(not self.online_prototypes, dtype=torch.bool))

    def initialize_prototypes(self, centers: torch.Tensor) -> None:
        if not self.online_prototypes:
            return
        expected = (self.num_classes, self.num_prototypes, self.prototypes.shape[-1])
        if tuple(centers.shape) != expected:
            raise ValueError(f"Prototype shape {tuple(centers.shape)} != {expected}")
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(centers.to(self.prototypes.device), dim=2))
            self.prototypes_initialized.fill_(True)

    def normalized_prototypes(self) -> torch.Tensor | None:
        if not self.online_prototypes or not bool(self.prototypes_initialized.item()):
            return None
        return F.normalize(self.prototypes, dim=2)

    def prototype_class_logits(self, projection: torch.Tensor) -> torch.Tensor | None:
        prototypes = self.normalized_prototypes()
        if prototypes is None:
            return None
        similarities = torch.einsum("bd,ckd->bck", projection, prototypes)
        return torch.logsumexp(similarities / self.prototype_temperature, dim=2)

    def prototype_diversity_loss(self, margin: float = 0.30) -> torch.Tensor:
        prototypes = self.normalized_prototypes()
        if prototypes is None or self.num_prototypes < 2:
            return self.classifier.weight.new_zeros(())
        similarities = torch.einsum("ckd,cmd->ckm", prototypes, prototypes)
        mask = ~torch.eye(self.num_prototypes, device=similarities.device, dtype=torch.bool)
        off_diagonal = similarities[:, mask].reshape(self.num_classes, -1)
        return F.relu(off_diagonal - float(margin)).pow(2).mean()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.adapter(x)
        projection = F.normalize(self.projector(features), dim=1)
        return {
            "logits": self.classifier(features),
            "projection": projection,
            "prototype_logits": self.prototype_class_logits(projection),
            "features": features,
        }


def build_model(config: dict, input_dim: int, num_classes: int) -> nn.Module:
    kind = config["model"]
    if kind == "linear":
        return LinearProbe(input_dim, num_classes)
    if kind == "mlp":
        return MLPClassifier(
            input_dim, int(config["hidden_dim"]), int(config["feature_dim"]), num_classes, float(config["dropout"])
        )
    if kind == "clean":
        return MetricProjector(
            input_dim,
            int(config["hidden_dim"]),
            int(config["feature_dim"]),
            int(config["projection_dim"]),
            float(config["dropout"]),
        )
    if kind == "hydrompc":
        return HydroMPC(
            input_dim,
            int(config["hidden_dim"]),
            int(config["feature_dim"]),
            int(config["projection_dim"]),
            num_classes,
            float(config["dropout"]),
        )
    if kind == "hydrompc_v2":
        return HydroMPCV2(
            input_dim,
            int(config["hidden_dim"]),
            int(config["feature_dim"]),
            int(config["projection_dim"]),
            num_classes,
            int(config["num_prototypes"]),
            float(config["prototype_temperature"]),
            float(config["dropout"]),
            bool(config.get("online_prototypes", False)),
        )
    raise ValueError(f"Unknown model type: {kind}")
