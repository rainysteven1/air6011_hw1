from config import cifar_config
from src.process import run
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIR6011 Homework1 Program")

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset",
        choices=["cifar", "mnist"],
        default="cifar",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="device",
        default="cuda:0",
    )
    args = parser.parse_args()

    config = None
    if args.dataset == "cifar":
        config = cifar_config
    config.device = args.device

    run(config)
