import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import get_densenet121
from dataset import IMAGENET_MEAN, IMAGENET_STD, IMG_HEIGHT, IMG_WIDTH


# Rutas relativas a la raíz del proyecto (mismo scope que el resto de scripts en src/).
MODEL_PATH = 'models/best_model_final_DenseNet121.pth'
IMAGE_PATH = 'data/train_images'   # se toma la primera imagen del directorio
RESULTS_DIR = 'results'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cargar_modelo():
    model = get_densenet121(pretrained=False)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def visualizar_filtros_primera_capa(model, output_path):
    # DenseNet-121 arranca con una Conv2D 7x7 con 64 filtros (shape: 64, 3, 7, 7).
    # Normalizamos cada filtro a [0,1] para verlo como una imagen RGB.
    primera_conv = model.model.features.conv0
    filtros = primera_conv.weight.data.cpu().numpy()

    n_filtros = filtros.shape[0]
    cols = 8
    rows = n_filtros // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for i, ax in enumerate(axes.flat):
        f = filtros[i].transpose(1, 2, 0)  # (7, 7, 3)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f)
        ax.axis('off')
        ax.set_title(f'{i}', fontsize=6)

    fig.suptitle(
        'Filtros aprendidos de la primera capa convolucional (DenseNet-121)\n'
        '64 filtros de tamaño 7x7x3. Cada uno detecta un patrón visual de bajo nivel.',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'OK: filtros guardados en {output_path}')


def cargar_imagen_prueba():
    archivos = sorted(os.listdir(IMAGE_PATH))[:1]
    if not archivos:
        raise FileNotFoundError(f'No hay imágenes en {IMAGE_PATH}')
    img_path = os.path.join(IMAGE_PATH, archivos[0])
    print(f'Imagen usada: {img_path}')

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Misma normalización que en entrenamiento.
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return img_resized, tensor.to(DEVICE)


def visualizar_feature_maps(model, output_path):
    # Hooks en 4 profundidades para capturar las activaciones al pasar una imagen.
    # De superficial a profunda: bordes/colores -> texturas -> partes -> conceptos.
    capas = {
        'conv0 (superficial)': model.model.features.conv0,
        'denseblock1': model.model.features.denseblock1,
        'denseblock3': model.model.features.denseblock3,
        'denseblock4 (profunda)': model.model.features.denseblock4,
    }

    activaciones = {}

    def make_hook(nombre):
        def hook(_module, _inp, out):
            activaciones[nombre] = out.detach().cpu()
        return hook

    handles = [capa.register_forward_hook(make_hook(n)) for n, capa in capas.items()]

    img_orig, tensor = cargar_imagen_prueba()
    with torch.no_grad():
        _ = model(tensor)

    for h in handles:
        h.remove()

    # Plot: imagen original + 8 mapas por cada una de las 4 capas.
    n_maps = 8
    n_filas = len(capas) + 1
    fig, axes = plt.subplots(n_filas, n_maps, figsize=(n_maps * 1.6, n_filas * 1.6))

    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title('Imagen original', fontsize=9)
    axes[0, 0].axis('off')
    for j in range(1, n_maps):
        axes[0, j].axis('off')

    for i, (nombre, _) in enumerate(capas.items(), start=1):
        act = activaciones[nombre][0]  # (C, H, W)
        for j in range(n_maps):
            mapa = act[j].numpy()
            axes[i, j].imshow(mapa, cmap='viridis')
            axes[i, j].axis('off')
        # Etiqueta a la izquierda manual (axis off oculta ylabel).
        axes[i, 0].text(
            -0.15, 0.5, nombre,
            transform=axes[i, 0].transAxes,
            rotation=90, ha='center', va='center', fontsize=9,
        )

    fig.suptitle(
        'Mapas de características en distintas profundidades de DenseNet-121\n'
        'Capas superficiales detectan bordes; capas profundas detectan patrones complejos.',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'OK: feature maps guardados en {output_path}')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f'Dispositivo: {DEVICE}')
    print('Cargando modelo...')
    model = cargar_modelo()

    visualizar_filtros_primera_capa(
        model, os.path.join(RESULTS_DIR, 'filtros_primera_capa.png')
    )
    visualizar_feature_maps(
        model, os.path.join(RESULTS_DIR, 'feature_maps.png')
    )
    print('Listo.')


if __name__ == '__main__':
    main()
