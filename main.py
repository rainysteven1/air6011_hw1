from config import cifar_config, mnist_config
from src.code_cupy.process import run as run_cupy
from src.code_torch.process import run as run_torch
from src.logger import logger
import argparse
import platform


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIR6011 Homework1 Program")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset",
        choices=["cifar", "mnist"],
        default="cifar",
    )

    parser.add_argument(
        "-l",
        "--library",
        type=str,
        help="device",
        choices=["cupy", "torch"],
        default="torch",
    )

    args = parser.parse_args()

    config = cifar_config if args.dataset == "cifar" else mnist_config
    logger.info(config)

    conf = {"running_platform": platform.node()}
    logger.info(conf)

    run_func = run_cupy if args.library == "cupy" else run_torch
    run_func(config)
