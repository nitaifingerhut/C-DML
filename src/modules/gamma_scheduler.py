import abc
import inspect
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

sns.set_style("darkgrid")
plt.rcParams["font.family"] = "serif"

from argparse import Namespace


class GammaScheduler:
    @classmethod
    def init_from_opts(cls, opts: Namespace):
        return cls(
            epochs=opts.sync_dml_epochs,
            warmup_epochs=opts.sync_dml_warmup_epochs,
            start_gamma=opts.sync_dml_start_gamma,
            end_gamma=opts.sync_dml_end_gamma,
        )

    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):
        # self.desc = "{} (epochs={}, warmup_epochs={}, start_gamma={:.3f}, end_gamma={:.3f})".format(
        #     self.__class__.__name__, epochs, warmup_epochs, start_gamma, end_gamma
        # )
        self.desc = "{} ({},{},{:.3f},{:.3f})".format(
            self.__class__.__name__, epochs, warmup_epochs, start_gamma, end_gamma
        )

    @abc.abstractmethod
    def __call__(self, epoch: int) -> float:
        raise NotImplementedError

    def __str__(self):
        return self.desc


class FixedGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):
        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        self.gamma = start_gamma
        self.desc = "{} (gamma={:.2f})".format(self.__class__.__name__, start_gamma)

    def __call__(self, epoch: int) -> float:
        return self.gamma


class LinearGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):
        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        self.gammas = np.linspace(start_gamma, end_gamma, epochs, endpoint=True, dtype=np.float32)
        self.desc = "{} (start_gamma={:.2f}, end_gamma={:.2f})".format(self.__class__.__name__, start_gamma, end_gamma)

    def __call__(self, epoch: int) -> float:
        return self.gammas[epoch]


class GeomGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 1e-6, end_gamma: float = 1.0):
        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        assert start_gamma > 0.0
        self.gammas = np.geomspace(start_gamma, end_gamma, epochs, endpoint=True, dtype=np.float32)
        self.desc = "{} (start_gamma={:.2f}, end_gamma={:.2f})".format(self.__class__.__name__, start_gamma, end_gamma)

    def __call__(self, epoch: int) -> float:
        return self.gammas[epoch]


class StepGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):

        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        self.start = start_gamma * np.ones(shape=(warmup_epochs,), dtype=np.float32,)
        self.end = end_gamma * np.ones(shape=(epochs - warmup_epochs,), dtype=np.float32,)
        self.gammas = np.concatenate((self.start, self.end))
        self.desc = "{} (warmup_epochs={}, start_gamma={:.2f}, end_gamma={:.2f})".format(
            self.__class__.__name__, warmup_epochs, start_gamma, end_gamma
        )

    def __call__(self, epoch: int) -> float:
        return self.gammas[epoch]


GAMMA_SCHEDULERS = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if obj.__module__ is __name__
}


# import pandas as pd
# if __name__ == "__main__":
#
#     nb_epochs = 10
#
#     schedulers = {
#         FixedGamma(epochs=nb_epochs, start_gamma=1.0),
#         LinearGamma(epochs=nb_epochs, start_gamma=0.0, end_gamma=1.0),
#         GeomGamma(epochs=nb_epochs, start_gamma=1e-6, end_gamma=1.0),
#         StepGamma(epochs=nb_epochs, warmup_epochs=nb_epochs // 2, start_gamma=0.1, end_gamma=0.9),
#     }
#
#     df = pd.DataFrame(columns=("scheduler", "epoch", "gamma"))
#
#     for epoch in range(nb_epochs):
#         for scheduler in schedulers:
#             df = pd.concat([df, pd.DataFrame({
#                 "scheduler": [scheduler.__class__.__name__], "epoch": epoch, "gamma": scheduler(epoch)
#             })], ignore_index=True)
#
#     _, ax = plt.subplots()
#     sns.lineplot(
#         data=df, x="epoch", y="gamma", hue="scheduler", markers=True, dashes=False, legend="auto"
#     )
#     ax.set_ylabel("$\\gamma$")
#     plt.tight_layout()
#     plt.show()
