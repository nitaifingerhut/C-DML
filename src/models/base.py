import abc
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from argparse import Namespace
from pathlib import Path
from sklearn.base import BaseEstimator
from torch import Tensor
from typing import Any, Dict, Tuple

sns.set_style("darkgrid")
plt.rcParams["font.family"] = "serif"


class BaseModel(BaseEstimator):

    HISTORY_KEYS = ("train loss", "train theta hat", "test loss", "test theta hat")

    @staticmethod
    @abc.abstractmethod
    def name():
        raise NotImplementedError

    def __init__(self, **kwargs):
        super().__init__()

        self.params = kwargs
        self.history = None
        self.reset_history()

    def __str__(self):
        return self.__class__.__name

    @classmethod
    @abc.abstractmethod
    def init_from_opts(cls, opts: Namespace, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def fit_params(self, opts: Namespace, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def restart(self):
        raise NotImplementedError

    def reset_history(self):
        self.history = {k: [] for k in self.HISTORY_KEYS}

    @abc.abstractmethod
    def fit(self, train: Dict[str, Tensor], test: Dict[str, Tensor], **kwargs):
        """
        Fit the model to the data.
        :param train: a dictionary with train data.
        :param test: a dictionary with test data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predicts (m(x), l(x)) for a given input x.
        :param x: a tensor of shape (num_samples, num_features).
        :return: m(x) and l(x), each of shape (num_samples, ).
        """
        raise NotImplementedError

    @staticmethod
    def _normalize_inputs(
        train: Dict[str, Tensor], test: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        y_range = train["y"].max() - train["y"].min()
        train["d"] = train["d"] / y_range
        train["y"] = train["y"] / y_range
        test["d"] = test["d"] / y_range
        test["y"] = test["y"] / y_range

        return train, test

    def _arange_stat(self, key):
        return np.arange(start=0, stop=len(self.history[key]))

    def plot_history(self, real_theta: float, save_as: Path):

        hist_copy = copy.deepcopy(self.history)

        num_train_entries = sum(["train" in x for x in hist_copy.keys()])
        num_test_entries = sum(["test" in x for x in hist_copy.keys()])
        assert num_train_entries == num_test_entries

        min_test_loss_index = hist_copy["test loss"].index(min(hist_copy["test loss"]))
        test_theta_hat = hist_copy["test theta hat"][min_test_loss_index]  # [-1]

        _, axs = plt.subplots(2, num_train_entries, figsize=(7.5 * num_train_entries / 2, 7.5))

        axs[0, 0].plot(self._arange_stat("train loss"), hist_copy["train loss"])
        axs[0, 0].set_ylabel("loss")
        axs[0, 0].set_title("train loss")
        axs[0, 0].set_yscale("log")
        del hist_copy["train loss"]

        axs[0, 1].plot(self._arange_stat("train theta hat"), hist_copy["train theta hat"])
        axs[0, 1].set_ylabel("$\hat{\\theta}$")
        axs[0, 1].set_title("train theta estimation")
        axs[0, 1].axhline(y=real_theta, color="red", alpha=0.8, linestyle="--")
        del hist_copy["train theta hat"]

        train_keys = {k: v for k, v in hist_copy.items() if "train" in k}
        for i, x in enumerate(train_keys.items()):
            k, v = x
            axs[0, i + 2].plot(self._arange_stat(k), v)
            axs[0, i + 2].set_title(k + "/{:.3}".format(hist_copy[k][min_test_loss_index]))

        axs[1, 0].plot(self._arange_stat("test loss"), hist_copy["test loss"])
        axs[1, 0].set_ylabel("loss")
        axs[1, 0].set_title("test loss")
        axs[1, 0].set_yscale("log")
        axs[1, 0].axvline(x=min_test_loss_index, color="red", alpha=0.8, linestyle="--")
        del hist_copy["test loss"]

        axs[1, 1].plot(self._arange_stat("test theta hat"), self.history["test theta hat"])
        axs[1, 1].set_ylabel("$\hat{\\theta}$")
        axs[1, 1].set_title("test theta estimation")
        axs[1, 1].axvline(x=min_test_loss_index, color="red", alpha=0.8, linestyle="--")
        axs[1, 1].axhline(y=real_theta, color="red", alpha=0.8, linestyle="--")
        del hist_copy["test theta hat"]

        test_keys = {k: v for k, v in hist_copy.items() if "test" in k}
        for i, x in enumerate(test_keys.items()):
            k, v = x
            axs[1, i + 2].plot(self._arange_stat(k), v)
            axs[1, i + 2].set_title(k + "/{:.3}".format(hist_copy[k][min_test_loss_index]))
            axs[1, i + 2].axvline(x=min_test_loss_index, color="red", alpha=0.8, linestyle="--")

        plt.suptitle(f"theta = {real_theta}, theta estimation = {round(test_theta_hat, 3)}")
        plt.subplots_adjust(wspace=0.45, hspace=0.25)
        plt.savefig(save_as, bbox_inches="tight")
        plt.close()
