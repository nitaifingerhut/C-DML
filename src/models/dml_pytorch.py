import tempfile
import torch

from argparse import Namespace
from src.models.base import BaseModel
from src.modules.nn_nets import Nets
from src.modules.optimizers import OPTIMIZERS, OPTIMIZERS_PARAMS
from torch import Tensor
from typing import Any, Dict, Tuple
from src.utils.estimator import est_theta_torch


class DoubleMachineLearningPyTorch(BaseModel):

    HISTORY_KEYS = (
        "train loss",
        "train theta hat",
        "train delta_m^2",
        "train delta_l^2",
        "test loss",
        "test theta hat",
        "test delta_m^2",
        "test delta_l^2",
    )

    @staticmethod
    def name():
        return "Double Machine Learning"

    def __init__(self, net_type: str, in_features: int, **kwargs):
        """
        :param net_type: Net type to use.
        :param num_features: Number of features in X.
        """
        super().__init__(**kwargs)

        self.net_type = net_type
        self.in_features = in_features
        self.net = Nets[net_type](in_features=in_features)

    @classmethod
    def init_from_opts(cls, opts: Namespace, **kwargs):
        return cls(net_type=opts.dml_net, in_features=opts.nb_features, **kwargs)

    def fit_params(self, opts: Namespace, **kwargs) -> Dict[str, Any]:
        return dict(
            learning_rate=opts.dml_lr,
            max_epochs=opts.dml_epochs,
            optimizer=opts.dml_optimizer,
            clip_grad_norm=opts.dml_clip_grad_norm,
        )

    def restart(self):
        self.net = Nets[self.net_type](in_features=self.in_features)
        self.reset_history()

    @staticmethod
    def _set_fit_params(**kwargs):
        return dict(
            learning_rate=kwargs.get("learning_rate", 0.001),
            max_epochs=kwargs.get("max_epochs", 1000),
            optimizer=kwargs.get("optimizer", "Adam"),
            clip_grad_norm=kwargs.get("clip_grad_norm", None),
        )

    def fit(self, train: Dict[str, Tensor], test: Dict[str, Tensor], **kwargs):
        """
        Fit the model to the data.
        :param train: a dictionary with train data.
        :param test: a dictionary with test data.
        """
        params = self._set_fit_params(**kwargs)

        loss_fn = torch.nn.MSELoss()
        optimizer_name = params["optimizer"]

        optimizer = OPTIMIZERS[optimizer_name](
            self.net.parameters(), lr=params["learning_rate"], **OPTIMIZERS_PARAMS[optimizer_name]
        )

        tmpfile = tempfile.NamedTemporaryFile(suffix=".pt")
        torch.save(self.net, tmpfile.name)

        test_min_loss = None
        for epoch in range(params["max_epochs"]):

            # Train
            optimizer.zero_grad()
            m_pred, l_pred = self.net(x=train["x"])
            m_loss = loss_fn(train["d"], m_pred)
            l_loss = loss_fn(train["y"], l_pred)
            loss = m_loss + l_loss
            loss.backward()

            if params["clip_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), params["clip_grad_norm"])
            optimizer.step()

            # Train statistics
            theta_hat, _ = est_theta_torch(train["y"].detach(), train["d"].detach(), m_pred.detach(), l_pred.detach())
            self.history["train loss"].append(loss.item())
            self.history["train theta hat"].append(theta_hat)
            self.history["train delta_m^2"].append(m_loss.item())
            self.history["train delta_l^2"].append(l_loss.item())

            # Evaluation
            with torch.no_grad():
                self.net.eval()
                m_hat, l_hat = self.net(x=test["x"])
                m_loss = loss_fn(test["d"], m_hat).item()
                l_loss = loss_fn(test["y"], l_hat).item()
                test_loss = m_loss + l_loss
                theta_hat, _ = est_theta_torch(
                    train["y"].detach(), train["d"].detach(), m_pred.detach(), l_pred.detach()
                )
                self.net.train()

            # Evaluation statistics
            self.history["test loss"].append(test_loss)
            self.history["test theta hat"].append(theta_hat)
            self.history["test delta_m^2"].append(m_loss)
            self.history["test delta_l^2"].append(l_loss)

            if test_min_loss is None or test_loss < test_min_loss:
                test_min_loss = test_loss
                torch.save(self.net, tmpfile.name)

        self.net = torch.load(tmpfile.name)
        tmpfile.close()

        return self

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predicts (m(x), l(x)) for a given input x.
        :param x: a tensor of shape (num_samples, num_features).
        :return: m(x) and l(x), each of shape (num_samples, ).
        """
        with torch.no_grad():
            m_hat, l_hat = self.net(x=x)
        return m_hat, l_hat
