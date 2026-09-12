import torch.nn as nn
import torchvision.models as models


def get_resnet_model(num_classes=4, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None

    model = models.resnet18(weights=weights)

    # Replace original ImageNet classifier with 4 output logits
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Keep final classifier trainable
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_backbone(model):
    # Fine-tune entire network
    for param in model.parameters():
        param.requires_grad = True