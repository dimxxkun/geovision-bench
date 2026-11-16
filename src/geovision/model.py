import torch
from torch import nn


class LandCoverCNN(nn.Module):
    """Small from-scratch RGB baseline; no pretrained weights or external imagery."""

    def __init__(self, classes=10, width=24):
        super().__init__()
        blocks = []
        previous = 3
        for channels in (width, width * 2, width * 4):
            blocks.extend(
                [
                    nn.Conv2d(previous, channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                ]
            )
            previous = channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(previous, classes)

    def forward(self, images):
        return self.classifier(self.features(images))


def preprocess(images):
    return images.to(torch.float32).div(255).sub(0.5).div(0.5)


def augment(images, generator):
    """Orientation augmentation applied only to training batches."""
    if torch.rand((), generator=generator).item() < 0.5:
        images = images.flip(-1)
    if torch.rand((), generator=generator).item() < 0.5:
        images = images.flip(-2)
    if torch.rand((), generator=generator).item() < 0.5:
        images = images.transpose(-1, -2)
    return images
