from argparse import Namespace
from src.models.base import BaseModel
from sklearn.linear_model import LinearRegression
from torch import Tensor
from typing import Any, Dict, Tuple


class DoubleMachineLearningSklearn(BaseModel):
    @staticmethod
    def name():
        return "Double Machine Learning"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_net = LinearRegression()
        self.l_net = LinearRegression()

    @classmethod
    def init_from_opts(cls, opts: Namespace, **kwargs):
        return cls(**kwargs)

    def fit_params(self, opts: Namespace, **kwargs) -> Dict[str, Any]:
        return {}

    def restart(self):
        self.m_net = LinearRegression()
        self.l_net = LinearRegression()
        self.reset_history()

    def fit(self, train: Dict[str, Tensor], test: Dict[str, Tensor], **kwargs):
        """
        Fit the model to the data.
        :param train: a dictionary with train data.
        :param test: a dictionary with test data.
        """
        self.m_net = self.m_net.fit(train["x"], train["d"])
        self.l_net = self.l_net.fit(train["x"], train["y"])
        return self

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predicts (m(x), l(x)) for a given input x.
        :param x: a tensor of shape (num_samples, num_features).
        :return: m(x) and l(x), each of shape (num_samples, ).
        """
        m_hat = self.m_net.predict(x)
        l_hat = self.l_net.predict(x)
        return m_hat, l_hat
