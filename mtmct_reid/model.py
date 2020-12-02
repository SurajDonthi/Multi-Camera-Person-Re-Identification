import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn import init
from torchvision.models import resnet50


def weights_init_kaiming(layer):
    if type(layer) in [nn.Conv1d, nn.Conv2d]:
        init.kaiming_normal_(layer.weight.data, mode='fan_in')
    elif type(layer) == nn.Linear:
        init.kaiming_normal_(layer.weight.data, mode='fan_out')
        init.constant_(layer.bias.data, 0.0)
    elif type(layer) == nn.BatchNorm1d:
        init.normal_(layer.weight.data, mean=1.0, std=0.02)
        init.constant_(layer.bias.data, 0.0)


def weights_init_classifier(layer):
    if type(layer) == nn.Linear:
        init.normal_(layer.weight.data, std=0.001)
        init.constant_(layer.bias.data, 0.0)


class ClassifierBlock(nn.Module):

    def __init__(self, input_dim: int, num_classes: int,
                 dropout: bool = True, activation: str = None,
                 num_bottleneck=512):
        super().__init__()
        self._layers(input_dim, num_classes, dropout,
                     activation, num_bottleneck)

    def _layers(self, input_dim, num_classes, dropout,
                activation, num_bottleneck):
        block = [
            nn.Linear(input_dim, num_bottleneck),
            nn.BatchNorm1d(num_bottleneck)
        ]
        if activation == 'relu':
            block += [nn.ReLU]
        elif activation == 'lrelu':
            block += [nn.LeakyReLU(0.1)]
        if dropout:
            block += [nn.Dropout(p=0.5)]
        block = nn.Sequential(*block)
        block.apply(weights_init_kaiming)

        classifier = nn.Linear(num_bottleneck, num_classes)
        classifier.apply(weights_init_classifier)

        self.block = block
        self.classifier = classifier

    def forward(self, x):
        x = self.block(x)
        x = self.classifier(x)
        return x


class PCB(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_parts = 6  # Parameterize?
        self._layers(num_classes)

    def _layers(self, num_classes):
        self.model = resnet50(pretrained=True)
        # Delete final fc layer
        del self.model.fc

        # Remove downsampling by changing stride to 1
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

        # Replace final layers
        self.model.avgpool = nn.AdaptiveAvgPool2d((self.num_parts, 1))
        self.dropout = nn.Dropout(p=0.5)

        self.model = nn.Sequential(*list(self.model.children()))

        # Define 6 separate layers for 6 parts!
        for i in range(self.num_parts):
            name = 'classifier'+str(i)
            setattr(self, name, ClassifierBlock(
                2048, num_classes, True, 'lrelu', 256))

    def forward(self, x, training=False):
        x = self.model(x)
        x = torch.squeeze(x)
        # Create a hook to identify whether
        # it is training phase or testing phase
        if training:
            x = self.dropout(x)
            part = []
            strips_out = []
            for i in range(self.num_parts):
                part.append(x[:, :, i])
                name = 'classifier' + str(i)
                classifier = getattr(self, name)
                part_out = classifier(part[i])
                strips_out.append(part_out)
            return strips_out

        return x    # return fc features
