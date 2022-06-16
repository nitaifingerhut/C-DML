import torch

from argparse import Namespace
from src.models.base import BaseModel
from sklearn.linear_model import LinearRegression
from torch import Tensor
from typing import Any, Dict


class OrdinaryLeastSquares(BaseModel):
    @staticmethod
    def name():
        return "Ordinary Least Squares"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear_regressor = LinearRegression(fit_intercept=False)

    @classmethod
    def init_from_opts(cls, opts: Namespace, **kwargs):
        return cls(**kwargs)

    def fit_params(self, opts: Namespace, **kwargs) -> Dict[str, Any]:
        return {}

    def restart(self):
        self.linear_regressor = LinearRegression(fit_intercept=False)
        self.reset_history()

    def fit(self, train: Dict[str, Tensor], test: Dict[str, Tensor], **kwargs):
        """
        Fit the model to the data.
        :param train: a dictionary with train data.
        :param test: a dictionary with test data.
        """
        x_tag = torch.cat((train["x"], train["d"].unsqueeze(1)), dim=1)
        self.linear_regressor = self.linear_regressor.fit(x_tag, train["y"])
        return self

    def theta(self):
        return self.linear_regressor.coef_[-1]
