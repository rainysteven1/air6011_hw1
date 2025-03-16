from omegaconf import ListConfig
from src.code_torch.model import Net
from src.dataeset import get_data_loader, get_transformations
from src.logger import logger
from src.plot import visualize
import os
import pandas as pd
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

    train_losses, test_accs = [], []
    for epoch in range(config.num_epochs):
        net.train()
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

        accuracy = correct / total
        train_losses.append(running_loss / len(loader["train"]))
        test_accs.append(accuracy)
        logger.info(
            f"Epoch [{epoch + 1}/{config.num_epochs}] "
            f"Train Loss: {train_losses[-1]:.3f} "
            f"Test Acc: {100 * accuracy:.2f}%"
        )

    logger.info("Finished Training")
    return train_losses, test_accs


def run(config: ListConfig):
    transformations = get_transformations()
    loader = get_data_loader(config.dataset, transformations)

    net = Net(**config.model)
    net.to(config.device)

    logger.info(
        f"number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    )

    train_losses, test_accs = _train(config.train, config.device, loader, net)
    visualize(train_losses, test_accs)

    torch.save(net.state_dict(), os.path.join(logger.log_dir, "model.pth"))

    df = pd.DataFrame(
        [
            {"Epoch": i, "Loss": loss, "Accuracy": acc}
            for i, (loss, acc) in enumerate(zip(train_losses, test_accs))
        ]
    )
    df.to_csv(os.path.join(logger.log_dir, "result.csv"), index=False)
