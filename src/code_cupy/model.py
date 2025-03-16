from src.code_cupy.module.function import ReLU
from src.code_cupy.module.base import Sequential
import src.code_cupy.module.layer as layer


class ConvBNReLU(Sequential):
    def __init__(self, input_dim, output_dim):
        super().__init__(
            layer.Conv2d(input_dim, output_dim, 3, padding=1),
            layer.BatchNorm2d(output_dim),
            ReLU(),
        )


class PooledDoubleConv(Sequential):
    def __init__(self, input_dim, output_dim, middle_dim=None):
        if not middle_dim:
            middle_dim = output_dim

        super().__init__(
            ConvBNReLU(input_dim, middle_dim),
            ConvBNReLU(middle_dim, output_dim),
            layer.MaxPool2d(2),
            layer.Dropout(0.25),
        )


class Net(Sequential):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(
            PooledDoubleConv(input_dim, 64),
            PooledDoubleConv(64, 128),
            PooledDoubleConv(128, 256),
            PooledDoubleConv(256, 512),
            layer.Flatten(),
            layer.Linear(512 * 1 * 1, 256),
            layer.BatchNorm1d(256),
            ReLU(),
            layer.Dropout(0.25),
            layer.Linear(256, output_dim),
        )
