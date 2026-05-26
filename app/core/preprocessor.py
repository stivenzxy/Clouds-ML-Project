import numpy as np
import torch
from PIL import Image


class ImagePreprocessor:
    # Pipeline IDÉNTICO al de validación en entrenamiento (resize + normalización ImageNet).

    def __init__(self, height: int, width: int, mean: tuple, std: tuple):
        self._height = height
        self._width = width
        self._mean = np.array(mean, dtype=np.float32)
        self._std = np.array(std, dtype=np.float32)

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        # Convierte PIL Image -> tensor (1, 3, H, W) listo para el modelo.
        image = image.convert('RGB').resize((self._width, self._height), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return tensor
