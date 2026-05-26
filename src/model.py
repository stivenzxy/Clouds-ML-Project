import torch.nn as nn
import timm


class CloudModel(nn.Module):
    # Devuelve LOGITS (sin sigmoid). BCEWithLogitsLoss aplica sigmoid internamente
    # de forma numéricamente estable. En inferencia: torch.sigmoid(model(x)).

    def __init__(self, model_name, pretrained=True, num_classes=4):
        super().__init__()
        # timm reemplaza la cabeza final por Linear con num_classes salidas.
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
