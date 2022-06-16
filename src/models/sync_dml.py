import tempfile
import torch

from argparse import Namespace
from src.models.base import BaseModel
from src.modules.gamma_scheduler import GAMMA_SCHEDULERS
from src.modules.loss import LOSSES
from src.modules.nn_nets import Nets
from src.modules.optimizers import OPTIMIZERS, OPTIMIZERS_PARAMS
from torch import Tensor
from typing import Any, Dict, Tuple
from src.utils.estimator import est_theta_torch


class SYNChronizedDoubleMachineLearning(BaseModel):

    HISTORY_KEYS = (
        "train loss",
        "train theta hat",
        "train residuals m.2",
        "train residuals l.2",
        "train residuals correlation",
        "test loss",
        "test theta hat",
        "test residuals m.2",
        "test residuals l.2",
        "test residuals correlation",
    )

    @staticmethod
    def name():
        return "Synchronized Double Machine Learning"

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
        return cls(net_type=opts.sync_dml_net, in_features=opts.nb_features, **kwargs)

    def fit_params(self, opts: Namespace, dml_stats: Dict[str, float], **kwargs) -> Dict[str, Any]:
        gamma_scheduler = GAMMA_SCHEDULERS[opts.sync_dml_gamma_scheduler].init_from_opts(opts)
        loss_fn = LOSSES[opts.sync_dml_loss](gamma_scheduler=gamma_scheduler, dml_stats=dml_stats)
        return dict(
            loss_fn=loss_fn,
            learning_rate=opts.sync_dml_lr,
            max_epochs=opts.sync_dml_epochs,
            optimizer=opts.sync_dml_optimizer,
            clip_grad_norm=opts.sync_dml_clip_grad_norm,
        )

    def restart(self):
        self.net = Nets[self.net_type](in_features=self.in_features)
        self.reset_history()

    def _set_fit_params(self, **kwargs):
        return dict(
            loss_fn=kwargs.get("loss_fn", None),
            learning_rate=kwargs.get("learning_rate", 0.0001),
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

        loss_fn = params["loss_fn"]
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
            m_hat, l_hat = self.net(x=train["x"])
            loss = loss_fn(train["d"], train["y"], m_hat, l_hat, epoch=epoch)
            loss.backward()

            if params["clip_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), params["clip_grad_norm"])
            optimizer.step()

            # Train statistics
            theta_hat, _ = est_theta_torch(train["y"].detach(), train["d"].detach(), m_hat.detach(), l_hat.detach())
            self.history["train loss"].append(loss.item())
            self.history["train theta hat"].append(theta_hat)

            res_m = train["d"].detach() - m_hat.detach()
            res_l = train["y"].detach() - l_hat.detach()
            self.history["train residuals m.2"].append(torch.mean(res_m ** 2).item())
            self.history["train residuals l.2"].append(torch.mean(res_l ** 2).item())
            self.history["train residuals correlation"].append(torch.mean(res_m * res_l).item())

            # Evaluation
            with torch.no_grad():
                self.net.eval()
                m_hat, l_hat = self.net(x=test["x"])
                test_loss = loss_fn(test["d"].detach(), test["y"].detach(), m_hat.detach(), l_hat.detach(), epoch=epoch)
                theta_hat, _ = est_theta_torch(test["y"].detach(), test["d"].detach(), m_hat.detach(), l_hat.detach())
                self.net.train()

            self.history["test loss"].append(test_loss.item())
            self.history["test theta hat"].append(theta_hat)

            res_m = test["d"].detach() - m_hat.detach()
            res_l = test["y"].detach() - l_hat.detach()
            self.history["test residuals m.2"].append(torch.mean(res_m ** 2).item())
            self.history["test residuals l.2"].append(torch.mean(res_l ** 2).item())
            self.history["test residuals correlation"].append(torch.mean(res_m * res_l).item())

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
