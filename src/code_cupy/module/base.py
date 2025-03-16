# base class
from itertools import chain


class Module:
    def __init__(self):
        self.input = None
        self._training = True

    @property
    def params(self):
        return []

    def forward(self, x):
        self.input = x

    def backward(self, d_output):
        pass

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Sequential(list):
    def __init__(self, *modules):
        super().__init__(modules)

    @property
    def params(self):
        return list(chain.from_iterable(m.params for m in self))

    def forward(self, x):
        y = x
        for module in self:
            y = module(y)
        return y

    def backward(self, dy):
        dx = dy
        for module in reversed(self):
            dx = module.backward(dx)
        return dx

    def train(self):
        for m in self:
            m.train()

    def eval(self):
        for m in self:
            m.eval()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
