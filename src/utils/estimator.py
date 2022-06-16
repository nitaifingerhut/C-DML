import numpy as np
import torch

from numpy import ndarray
from torch import Tensor
from typing import Tuple, Union


def est_theta_numpy(y: ndarray, d: ndarray, m_hat: ndarray, l_hat: ndarray) -> Tuple[float, float]:
    v_hat = d - m_hat
    mean_v_hat_2 = np.mean(v_hat * v_hat)
    theta_hat = np.mean(v_hat * (y - l_hat)) / mean_v_hat_2
    return theta_hat.item(), mean_v_hat_2.item()


def est_theta_torch(y: Tensor, d: Tensor, m_hat: Tensor, l_hat: Tensor) -> Tuple[float, float]:
    v_hat = d - m_hat
    mean_v_hat_2 = torch.mean(v_hat * v_hat)
    theta_hat = torch.mean(v_hat * (y - l_hat)) / mean_v_hat_2
    return theta_hat.item(), mean_v_hat_2.item()


def pearson_correlation_numpy(x: np.ndarray, y: np.ndarray):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    coeff = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return coeff


def pearson_correlation_torch(x: Tensor, y: Tensor):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    coeff = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return coeff.item()


def est_theta(
    y: Union[ndarray, Tensor], d: Union[ndarray, Tensor], m_hat: Union[ndarray, Tensor], l_hat: Union[ndarray, Tensor]
) -> Tuple[float, float]:
    if all(isinstance(i, Tensor) for i in (y, d, m_hat, l_hat)):
        return est_theta_torch(y, d, m_hat, l_hat)
    if all(isinstance(i, ndarray) for i in (y, d, m_hat, l_hat)):
        return est_theta_numpy(y, d, m_hat, l_hat)
    raise TypeError


def pearson_correlation(x: Union[ndarray, Tensor], y: Union[ndarray, Tensor]) -> Tuple[float, float]:
    if all(isinstance(i, Tensor) for i in (x, y)):
        return pearson_correlation_torch(x, y)
    if all(isinstance(i, ndarray) for i in (x, y)):
        return pearson_correlation_numpy(x, y)
    raise TypeError
