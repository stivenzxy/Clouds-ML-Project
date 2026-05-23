import torch
import torch.nn as nn
import timm

class CloudModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(CloudModel, self).__init__()
        
        # Cargar el modelo base desde timm con 4 clases de salida
        # Esto reemplaza automáticamente la cabeza final por una Linear de 4 salidas
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=4)
        
        # Añadir activación Sigmoid para multi-label
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # El modelo de timm ya incluye la nueva capa lineal de 4 salidas
        # Retornamos logits directamente para BCEWithLogitsLoss
        x = self.model(x)
        return x

def get_densenet121(pretrained=True):
    return CloudModel('densenet121', pretrained=pretrained)

def get_mobilenetv3_large(pretrained=True):
    return CloudModel('mobilenetv3_large_100', pretrained=pretrained)
