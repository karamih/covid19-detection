import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# from torchinfo import summary


def model(out_features: int):
    weights = ResNet18_Weights.DEFAULT
    resnet_model = resnet18(weights=weights)

    for params in resnet_model.parameters():
        params.requires_grad = False
    for params in resnet_model.fc.parameters():
        params.requires_grad = True

    resnet_model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=out_features))

    # print(summary(resnet_model,
    #               input_size=[1, 3, 224, 224],
    #               col_names=['input_size', 'output_size', 'num_params', 'trainable'],
    #               col_width=20))

    return resnet_model