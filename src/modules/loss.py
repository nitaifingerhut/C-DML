import inspect
import sys
import torch

from src.modules.gamma_scheduler import GammaScheduler
from torch import Tensor
from typing import Dict


class CorrelationLoss:
    def __init__(self, gamma_scheduler: GammaScheduler, dml_stats: Dict[str, float]):
        self.gamma_scheduler = gamma_scheduler
        self.corr_abs = dml_stats["corr.abs"]
        self.res_m_2 = dml_stats["res_m.2"]
        self.res_l_2 = dml_stats["res_l.2"]

    def __call__(self, d: Tensor, y: Tensor, m_hat: Tensor, l_hat: Tensor, **kwargs):

        ### Residuals
        res_m = d - m_hat
        res_l = y - l_hat

        ### Reconstruction
        res_m_2 = torch.mean(res_m ** 2)
        res_l_2 = torch.mean(res_l ** 2)
        res_corr_abs = torch.absolute(torch.mean(res_m * res_l))

        ### Normalization
        res_m_2 = res_m_2 / self.res_m_2
        res_l_2 = res_l_2 / self.res_l_2
        res_corr_abs = res_corr_abs / self.corr_abs

        ## Loss
        gamma = self.gamma_scheduler(**kwargs)
        loss = res_m_2 + res_l_2 + gamma * res_corr_abs
        return loss


LOSSES = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if obj.__module__ is __name__
}
