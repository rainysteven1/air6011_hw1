from omegaconf import ListConfig
from src.code_cupy.model import Net
from src.dataeset import get_data_loader, get_transformations
from src.logger import logger
from src.plot import visualize
import cupy as cp
import os
import pandas as pd
import src.code_cupy.module as mo
import src.code_cupy.optimizer as optim

__all__ = ["run"]


def _train(config: ListConfig, device: str, loader, net: mo.Module):
    optimizer = getattr(optim, config.optim.name)(net.params, **config.optim.params)

    criterion = mo.CrossEntropyLoss()

    train_losses, test_accs = [], []
    for epoch in range(config.num_epochs):
        net.train()
        running_loss = 0.0
        for i, (img, target) in enumerate(loader["train"]):
            img = cp.asarray(img.to(device))
            target = cp.asarray(target.to(device))

            pred = net(img)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            net.backward(criterion.backward(loss))
            optimizer.step()

            running_loss += loss.item()
            if i % config.print_every == config.print_every - 1:
                message = f"[epoch={epoch + 1:3d}, iter={i + 1:5d}] loss: {running_loss / config.print_every:.3f}"
                logger.info(message)
                running_loss = 0.0

        net.eval()
        correct, total = 0, 0
        for img, target in loader["test"]:
            img = cp.asarray(img.to(device))
            target = cp.asarray(target.to(device))

            pred = net(img)

            total += len(target)
            correct += (cp.argmax(pred, axis=1) == target).sum().item()

        accuracy = correct / total
        train_losses.append(running_loss / len(loader["train"]))
        test_accs.append(accuracy)
        logger.info(
            f"Epoch [{epoch + 1}/{epoch}] "
            f"Train Loss: {train_losses[-1]:.3f} "
            f"Test Acc: {100 * accuracy:.2f}%"
        )

    logger.info("Finished Training")
    return train_losses, test_accs


def run(config: ListConfig):
    if config.device == "cuda":
        cp.cuda.Device(0).use()

    transformations = get_transformations()
    loader = get_data_loader(config.dataset, transformations)

    net = Net(**config.model)

    train_losses, test_accs = _train(config.train, config.device, loader, net)
    visualize(train_losses, test_accs)

    cp.save(
        os.path.join(logger.log_dir, "model.npy"),
        [cp.asnumpy(p["val"]) for p in net.params],
    )

    df = pd.DataFrame(
        [
            {"Epoch": i, "Loss": loss, "Accuracy": acc}
            for i, (loss, acc) in enumerate(zip(train_losses, test_accs))
        ]
    )
    df.to_csv(os.path.join(logger.log_dir, "result.csv"), index=False)
