from src.code_torch.model import Net
from src.dataeset import get_data_loader, get_transformations
from src.logger import logger
from omegaconf import ListConfig
import os
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["run"]


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
    transformations = get_transformations()
    loader = get_data_loader(config.dataset, transformations)

    net = Net(**config.model)
    net.to(config.device)

    message = f"number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    logger.info(message)

    _train(config.train, config.device, loader, net)
    _test(config.device, loader, net)

    torch.save(net.state_dict(), os.path.join(logger.log_dir, "model.pth"))
