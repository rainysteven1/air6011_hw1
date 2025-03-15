from typing import Dict
from src.logger import logger
from src.model import Net
from omegaconf import ListConfig
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

__all__ = ["run"]


def _get_transformation():
    transformation = {}
    for data_type in ("train", "test"):
        is_train = data_type == "train"
        transformation[data_type] = tv_transforms.Compose(
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

    return transformation


def _get_data_loader(config: ListConfig, transformation: Dict):
    dataset, loader = {}, {}
    for data_type in ("train", "test"):
        is_train = data_type == "train"
        className = tv_datasets.CIFAR10 if config.name == "cifar" else tv_datasets.MNIST
        dataset[data_type] = className(
            root="./data",
            train=is_train,
            download=False,
            transform=transformation[data_type],
        )
        loader[data_type] = torch.utils.data.DataLoader(
            dataset[data_type],
            batch_size=config.batch_size,
            shuffle=is_train,
            num_workers=config.num_workers,
        )

    return loader


def _train(config: ListConfig, device: str, loader, net: nn.Module):
    # the network optimizer
    optimizer = getattr(optim, config.optim.name)(
        net.parameters(), **config.optim.params
    )

    # loss function
    criterion = nn.CrossEntropyLoss()

    net.train()

    for epoch in range(config.num_epochs):
        running_loss = 0.0
        for i, (img, target) in enumerate(loader["train"]):
            img, target = img.to(device), target.to(device)

            pred = net(img)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % config.print_every == config.print_every - 1:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "metrics": {...},
                }
                torch.save(checkpoint, os.path.join(logger.log_dir, "checkpoint.pth"))

                message = f"[epoch={epoch + 1:3d}, iter={i + 1:5d}] loss: {running_loss / config.print_every:.3f}"
                logger.info(message)
                running_loss = 0.0

    logger.info("Finished Training")


def _test(device, loader, net: nn.Module):
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for img, target in loader["test"]:
            img, target = img.to(device), target.to(device)

            # make prediction
            pred = net(img)

            # accumulate
            total += len(target)
            correct += (torch.argmax(pred, dim=1) == target).sum().item()

    message = f"Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%"
    logger.info(message)

    logger.info("Finished Testing")


def run(config: ListConfig):
    logger.info(config)

    transformation = _get_transformation()
    loader = _get_data_loader(config.dataset, transformation)

    net = Net(**config.model)
    net.to(config.device)

    message = f"number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    logger.info(message)

    _train(config.train, config.device, loader, net)
    _test(config.device, loader, net)

    torch.save(net.state_dict(), os.path.join(logger.log_dir, "model.pth"))
