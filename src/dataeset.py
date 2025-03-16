from typing import Dict
from omegaconf import ListConfig
import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms


def get_transformations():
    transformations = {"cifar": {}, "mnist": {}}
    for data_type in ("train", "test"):
        is_train = data_type == "train"
        transformations["cifar"][data_type] = tv_transforms.Compose(
            (
                [
                    tv_transforms.RandomCrop(32, padding=4),
                    tv_transforms.RandomHorizontalFlip(p=0.5),
                    tv_transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    tv_transforms.RandAugment(),
                    tv_transforms.RandomRotation(degrees=15),
                    tv_transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                ]
                if is_train
                else []
            )
            + [
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                ),
            ]
        )

        transformations["mnist"][data_type] = tv_transforms.Compose(
            (
                [
                    tv_transforms.RandomCrop(28, padding=4),
                    tv_transforms.RandomAffine(
                        degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    tv_transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ]
                if is_train
                else []
            )
            + [
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(
                    mean=[
                        0.1307,
                    ],
                    std=[
                        0.3801,
                    ],
                ),
            ]
        )

    return transformations


def get_data_loader(config: ListConfig, transformations: Dict[str, Dict]):
    dataset, loader = {}, {}
    for data_type in ("train", "test"):
        is_train = data_type == "train"
        className = tv_datasets.CIFAR10 if config.name == "cifar" else tv_datasets.MNIST
        dataset[data_type] = className(
            root="./data",
            train=is_train,
            download=config.download,
            transform=transformations[config.name][data_type],
        )
        loader[data_type] = torch.utils.data.DataLoader(
            dataset[data_type],
            batch_size=config.batch_size,
            shuffle=is_train,
            num_workers=config.num_workers,
        )

    return loader
