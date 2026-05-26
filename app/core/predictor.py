from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class Prediction:
    probabilities: Dict[str, float]

    def labels_at(self, threshold: float) -> Tuple[str, ...]:
        return tuple(name for name, p in self.probabilities.items() if p >= threshold)

    def top_class(self) -> Tuple[str, float]:
        name, prob = max(self.probabilities.items(), key=lambda kv: kv[1])
        return name, prob


class CloudPredictor:
    def __init__(
        self,
        model: nn.Module,
        class_names: Tuple[str, ...],
        device: torch.device,
    ):
        self._model = model.to(device).eval()
        self._class_names = class_names
        self._device = device

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> Prediction:
        # Modelo devuelve logits; aplicamos sigmoid para obtener probabilidades en [0, 1].
        tensor = tensor.to(self._device)
        logits = self._model(tensor)
        probs = torch.sigmoid(logits).cpu().squeeze(0).numpy()
        probabilities = {
            name: float(p) for name, p in zip(self._class_names, probs)
        }
        return Prediction(probabilities=probabilities)
