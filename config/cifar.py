from omegaconf import OmegaConf
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    name: str = "cifar"
    batch_size: int = 64
    num_workers: int = 2


@dataclass
class ModelConfig:
    input_dim: int = 3
    output_dim: int = 10


@dataclass
class OptimParams:
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4


@dataclass
class OptimConfig:
    name: str = "SGD"
    params: OptimParams = field(default_factory=OptimParams)


@dataclass
class TrainConfig:
    optim: OptimConfig = field(default_factory=OptimConfig)
    num_epochs: int = 128
    print_every: int = 200


config = OmegaConf.structured(
    {
        "dataset": DatasetConfig(),
        "device": "cuda",
        "model": ModelConfig(),
        "train": TrainConfig(),
    }
)
