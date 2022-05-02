import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


EXTRAS = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
}
MBOX = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.out_channels = out_channels
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        # x = self.relu(x)
        return x


class ConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # self.res = in_channels == out_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward().requires_grad_(False)
        self.size = size

        # SSD network, use mobilenetv1
        self.backbone = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBN(3, 64, 1),
                    ConvDW(64, 128, 2),
                    ConvDW(128, 128, 1),
                    ConvDW(128, 256, 2),
                    ConvDW(256, 256, 1),
                    ConvDW(256, 512, 2),
                ),
                nn.Sequential(
                    ConvDW(512, 512, 1),
                    ConvDW(512, 1024, 2),
                    ConvDW(1024, 1024, 1),
                ),
            ]
        )

        extras_layers, loc_layers, conf_layers = self.multibox(
            add_extras(EXTRAS[str(size)], 1024),
            MBOX[str(size)], num_classes
        )
        self.extras = nn.ModuleList(extras_layers)
        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)

        if phase == 'test':
            self.detect = Detect(num_classes, 0, 10000, 0.01, 0.45)

    def multibox(self, extra_layers, cfg, num_classes):
        loc_layers = []
        conf_layers = []

        for i, module in enumerate(self.backbone):
            loc_layers.append(nn.Conv2d(module[-1].out_channels, cfg[i] * 4, kernel_size=3, padding=1, bias=False))
            conf_layers.append(nn.Conv2d(module[-1].out_channels, cfg[i] * num_classes, kernel_size=3, padding=1, bias=False))

        for i, module in enumerate(extra_layers[1::2], 2):
            loc_layers.append(nn.Conv2d(module.out_channels, cfg[i] * 4, kernel_size=3, padding=1, bias=False))
            conf_layers.append(nn.Conv2d(module.out_channels, cfg[i] * num_classes, kernel_size=3, padding=1, bias=False))

        return extra_layers, loc_layers, conf_layers

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch * num_priors,num_classes]
                    2: localization layers, Shape: [batch, num_priors * 4]
                    3: priorbox layers, Shape: [2, num_priors * 4]
        """
        features = []
        loc = []
        conf = []

        for layer in self.backbone:
            x = layer(x)
            features.append(x)

        # apply extra layers and cache feature layer outputs
        for i, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if i % 2 == 1:
                features.append(x)

        # apply multibox head to source layers
        for (f, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(f).permute(0, 2, 3, 1).contiguous())
            conf.append(c(f).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect.apply(
                21, 0, 10000, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),      # loc preds
                torch.softmax(conf.view(conf.size(0), -1, self.num_classes), dim=-1),   # conf preds
                self.priors.to(x.device)           # default boxes
            )
            return output
        elif self.phase == "convert":
            return loc, conf
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors.to(x.device)
            )
            return output

    def load_weights(self, base_file):
        _, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_extras(cfg, in_channels, flag=False):
    # Extra layers for feature scaling
    layers = []
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1, bias=False)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag], bias=False)]
            flag = not flag
        in_channels = v
    return layers


def build_ssd(phase, size=300, num_classes=21):
    return SSD(phase, size, num_classes)


if __name__ == "__main__":
    ssd = build_ssd("train")
    dummy_input = torch.rand((2, 3, 416, 416))
    dummy_output = ssd(dummy_input)
    for o in dummy_output:
        print(o.shape)
