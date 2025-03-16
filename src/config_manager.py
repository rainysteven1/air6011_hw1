from dataclasses import dataclass
from omegaconf import OmegaConf
from pathlib import Path
import json


@dataclass
class DatasetConfig:
    name: str
    download: bool
    batch_size: int
    num_workers: int


@dataclass
class ModelConfig:
    input_dim: int
    output_dim: int


@dataclass
class OptimParams:
    lr: float


@dataclass
class OptimConfig:
    name: str
    params: OptimParams


@dataclass
class TrainConfig:
    optim: OptimConfig
    num_epochs: int
    print_every: int


class ConfigManager:
    @staticmethod
    def save(config, path: str):
        """保存配置到JSON文件"""
        config_dict = OmegaConf.to_container(config, resolve=True)

        # 序列化并保存
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4)

        # 创建关联的OmegaConf元文件
        meta_path = Path(path).with_suffix(".meta.yaml")
        OmegaConf.save(config, meta_path)

    @classmethod
    def load(cls, path: str):
        """从JSON文件加载配置"""
        with open(path, "r") as f:
            config_dict = json.load(f)

        return OmegaConf.structured(
            {
                "dataset": DatasetConfig(**config_dict.get("dataset", {})),
                "device": config_dict.get("device", "cuda"),
                "model": ModelConfig(**config_dict.get("model", {})),
                "train": TrainConfig(
                    **{k: v for k, v in config_dict["train"].items() if k != "optim"},
                    optim=OptimConfig(
                        name=config_dict["train"]["optim"]["name"],
                        params=OptimParams(**config_dict["train"]["optim"]["params"]),
                    ),
                ),
            }
        )
