import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pdb
from resnet import *

class DCLModel(nn.Module):
    def __init__(self, config):
        super(DCLModel, self).__init__()
        self.num_classes = config.numcls
        print(self.backbone_arch)

        self.model=resnet50(pretrained=True)

        self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.classifier_swap = nn.Linear(2048, 2 * self.num_classes, bias=False)
        self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=True)
        self.avgpool2 = nn.AvgPool2d(2, stride=2)


    def forward(self, x):
        x = self.model(x)
        mask = self.Convmask(x)
        mask = self.avgpool2(mask)
        mask = torch.tanh(mask)
        mask = mask.view(mask.size(0), -1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = []
        out.append(self.classifier(x))
        out.append(self.classifier_swap(x))
        out.append(mask)

        return out
