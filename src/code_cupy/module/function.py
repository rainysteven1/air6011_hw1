from src.code_cupy.module.base import Module
import cupy as cp


class LeakyRelu(Module):
    def __init__(self, negative_slope: float):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        super().forward()
        return cp.where(x > 0, x, self.alpha * x)

    def backward(self, d_output):
        return d_output * cp.where(self.input > 0, 1.0, self.alpha)


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input = x > 0
        return cp.maximum(x, 0)

    def backward(self, d_output):
        return d_output * self.input.astype(cp.float32)


class SELU(Module):
    def __init__(self, alpha: float, scale: float):
        super().__init__()
        self.alpha = alpha
        self.scale = scale

    def forward(self, x):
        super().forward()
        return self.scale * (
            cp.where(x > 0, x, 0.0)
            + self.alpha * (cp.exp(cp.where(x <= 0, x, 0.0)) - 1)
        )

    def backward(self, d_output):
        return (
            d_output
            * cp.where(self.input > 0, 1.0, self.alpha * cp.exp(self.input))
            * self.scale
        )


class Sigmoid(Module):
    def __init__(self):
        super.__init__()

    def forward(self, x):
        super().forward()
        return 1 / (1 + cp.exp(-x))

    def backward(self, d_output):
        sig = 1 / (1 + cp.exp(-self.input))
        return d_output * sig * (1 - sig)
