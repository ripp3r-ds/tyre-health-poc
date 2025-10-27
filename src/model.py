import torch
from torchvision import models
from typing import Optional

def build_model(num_classes: int, pretrained: bool = True, dropout: Optional[float] = 0.5):
    m = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )
    # Freeze all
    for p in m.parameters():
        p.requires_grad = False
    # Unfreeze final block for a bit of capacity
    for p in m.layer4.parameters():
        p.requires_grad = True

    in_feats = m.fc.in_features
    if dropout and dropout > 0:
        m.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_feats, num_classes)
        )
    else:
        m.fc = torch.nn.Linear(in_feats, num_classes)
    return m
