from omegaconf import ListConfig
from src.code_cupy.model import Net
from src.dataeset import get_data_loader, get_transformations
from src.logger import logger
import cupy as cp
import src.code_cupy.module as mo
import src.code_cupy.optimizer as optim


def _train(config: ListConfig, device: str, loader, net: mo.Module):
    optimizer = getattr(optim, config.optim.name)(net.params, **config.optim.params)

    criterion = mo.CrossEntropyLoss()

    net.train()

    for epoch in range(config.num_epochs):
        running_loss = 0.0
        for i, (img, target) in enumerate(loader["train"]):
            img = cp.asarray(img.to(device))
            target = cp.asarray(target.to(device))

            pred = net(img)

            loss = criterion(pred, target)

            optimizer.zero_grad()
            net.backward(criterion.backward(loss))
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % config.print_every == config.print_every - 1:
                message = f"[epoch={epoch + 1:3d}, iter={i + 1:5d}] loss: {running_loss / config.print_every:.3f}"
                logger.info(message)
                running_loss = 0.0

    logger.info("Finished Training")


def _test(device, loader, net: mo.Module):
    net.eval()

    correct, total = 0, 0
    for img, target in loader["test"]:
        img = cp.asarray(img.to(device))
        target = cp.asarray(target.to(device))

        pred = net(img)

        total += len(target)
        correct += (cp.argmax(pred, axis=1) == target).sum().item()

    message = f"Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%"
    logger.info(message)

    logger.info("Finished Testing")


def run(config: ListConfig):
    if config.device == "cuda":
        cp.cuda.Device(0).use()

    transformations = get_transformations()
    loader = get_data_loader(config.dataset, transformations)

    net = Net(**config.model)

    # message = f"number of parameters: {sum(p.numel() for p in net.params.val if p.requires_grad) / 1_000_000:.2f}M"
    # logger.info(message)

    _train(config.train, config.device, loader, net)
    _test(config.device, loader, net)
