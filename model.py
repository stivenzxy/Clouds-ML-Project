import torch.nn as nn
import timm


class CloudModel(nn.Module):
    """
    Wrapper sobre timm para clasificación multi-etiqueta de nubes.

    Devuelve LOGITS crudos (sin sigmoid), porque la pérdida usada es
    BCEWithLogitsLoss, que aplica sigmoid internamente de forma estable
    numéricamente. Si en inferencia se necesitan probabilidades, aplicar
    torch.sigmoid(model(x)) externamente.
    """

    def __init__(self, model_name, pretrained=True, num_classes=4):
        super().__init__()
        # timm reemplaza automáticamente la cabeza final por una Linear
        # con `num_classes` salidas (4 = Fish, Flower, Gravel, Sugar).
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


def get_densenet121(pretrained=True):
    return CloudModel('densenet121', pretrained=pretrained)


def get_mobilenetv3_large(pretrained=True):
    return CloudModel('mobilenetv3_large_100', pretrained=pretrained)
