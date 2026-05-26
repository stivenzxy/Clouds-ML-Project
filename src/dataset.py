import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Tamaño objetivo: relación 2:3 conservada (1400x2100 -> 320x480), divisible por 32.
IMG_HEIGHT = 320
IMG_WIDTH = 480

# Estadísticas de ImageNet, obligatorias para los modelos preentrenados.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CloudDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = self.df['image_id'].values
        self.labels = self.df[['Fish', 'Flower', 'Gravel', 'Sugar']].values.astype('float32')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        return image, torch.from_numpy(label)


def get_train_transforms():
    # Resize PRIMERO: aplicar augmentations sobre 1400x2100 es ~20x más caro.
    return A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.90, 1.10),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_valid_transforms():
    # Validación y test: sin augmentations, solo resize + normalización.
    return A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    train_df = pd.read_csv('data/train_split.csv')
    train_dir = 'data/train_images'

    dataset = CloudDataset(train_df, train_dir, transforms=get_train_transforms())
    img, label = dataset[0]
    print(f"Imagen shape: {img.shape}")
    print(f"Etiquetas: {label}")
