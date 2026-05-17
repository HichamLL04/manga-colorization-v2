import torch
import torch.nn as nn
import torchvision.models as M

from networks.models import SpectralNorm, SpectrResNeXtBottleneck


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.feed = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(64, 64, 3, 2, 0)),
            nn.LeakyReLU(0.2, True),
            SpectrResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1),
            SpectrResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1, stride=2),
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.LeakyReLU(0.2, True),
            SpectrResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1),
            SpectrResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1, stride=2),
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.LeakyReLU(0.2, True),
            SpectrResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
            SpectrResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1, stride=2),
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=1, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
            SpectrResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            SpectrResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),
            SpectrResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            SpectrResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out = nn.Linear(512, 1)

    def forward(self, color):
        x = self.feed(color)
        return self.out(x.view(color.size(0), -1))


class VGGContentLoss(nn.Module):
    def __init__(self, weights_path=None):
        super(VGGContentLoss, self).__init__()
        vgg16 = M.vgg16(weights=None)
        if weights_path:
            vgg16.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.features = nn.Sequential(*list(vgg16.features.children())[:9])
        for param in self.features.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.FloatTensor([0.485 - 0.5, 0.456 - 0.5, 0.406 - 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        return self.features((images.mul(0.5) - self.mean) / self.std)


def white_color_penalty(fake, target):
    mask = (~((target > 0.85).float().sum(dim=1) == 3).unsqueeze(1).repeat((1, 3, 1, 1))).float()
    white_zones = mask * (fake + 1) / 2
    return (torch.pow(white_zones.sum(dim=1), 2).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2, 3)) + 1)).mean()


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv2d') != -1 and hasattr(module, 'weight'):
        nn.init.xavier_uniform_(module.weight.data)


def weights_init_spectral(module):
    classname = module.__class__.__name__
    if classname.find('Conv2d') != -1 and hasattr(module, 'weight_bar'):
        nn.init.xavier_uniform_(module.weight_bar.data)

