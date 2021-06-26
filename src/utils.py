import numpy as np
import torch


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the off-diagonal elements of a square matrix

    References:
        https://github.com/facebookresearch/barlowtwins/blob/main/main.py#L180
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def round_to_nearest_odd(n: float) -> int:
    """Round up to an odd number (for Gaussian blur filter size)"""
    return int(np.ceil(n) // 2 * 2 + 1)
