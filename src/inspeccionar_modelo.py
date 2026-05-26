import torch.nn as nn
from model import get_densenet121


def main():
    model = get_densenet121(pretrained=False)

    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    n_bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    n_params = sum(p.numel() for p in model.parameters())

    print('=' * 60)
    print('Arquitectura DenseNet-121')
    print('=' * 60)
    print(f'Capas convolucionales (Conv2d): {n_conv}')
    print(f'Capas lineales (Linear):        {n_linear}')
    print(f'Capas BatchNorm2d:              {n_bn}')
    print(f'Parámetros totales:             {n_params:,}')
    print()
    print('Estructura de alto nivel:')
    print('-' * 60)
    for nombre, _ in model.model.features.named_children():
        print(f'  features.{nombre}')
    print()
    print('Detalle del primer bloque denso (denseblock1):')
    print('-' * 60)
    for nombre, _ in model.model.features.denseblock1.named_children():
        print(f'  denseblock1.{nombre}')


if __name__ == '__main__':
    main()
