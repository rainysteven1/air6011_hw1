import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, input_dim, output_dim):
        super().__init__(
            nn.Conv2d(input_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )


class PooledDoubleConv(nn.Sequential):
    def __init__(self, input_dim, output_dim, middle_dim=None):
        if not middle_dim:
            middle_dim = output_dim

        super().__init__(
            ConvBNReLU(input_dim, middle_dim),
            ConvBNReLU(middle_dim, output_dim),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )


class Net(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(
            nn.Sequential(
                PooledDoubleConv(input_dim, 64),
                PooledDoubleConv(64, 128),
                PooledDoubleConv(128, 256),
                PooledDoubleConv(256, 512),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(512 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, output_dim),
            )
        )
