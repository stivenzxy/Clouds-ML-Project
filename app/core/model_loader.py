import sys
from pathlib import Path

import torch
from torch import nn

# Reusa la definición de arquitectura de src/model.py (compartida con entrenamiento).
_SRC_PATH = Path(__file__).resolve().parent.parent.parent / 'src'
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from model import get_densenet121  # noqa: E402


def load_densenet121(weights_path: Path, device: torch.device) -> nn.Module:
    # pretrained=False: los pesos de ImageNet se sustituyen por los entrenados sobre nubes.
    # weights_only=True: evita el warning de seguridad de torch.load.
    model = get_densenet121(pretrained=False)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model
