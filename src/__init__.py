import cupy as cp
import numpy as np
import random
import torch


def set_seed(seed: int = 42):
    """Set all random seeds to ensure repeatable experiments"""
    cp.random.seed(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed()
