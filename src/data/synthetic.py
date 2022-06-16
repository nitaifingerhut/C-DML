import numpy as np
import torch

from functools import reduce
from operator import mul
from sklearn.model_selection import train_test_split
from typing import Tuple


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gaussian(z, sigma=1.0):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z / (sigma ** 2))


def m0_g0_s1(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = x[majority, 1] + 10 * x[majority, 3] + 5 * x[majority, 6]
    m0[minority] = 10 * x[minority, 1] + x[minority, 3] + 5 * x[minority, 6]

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = x[majority, 0] + 10 * x[majority, 2] + 5 * x[majority, 5]
    g0[minority] = 10 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5]

    return m0, g0


def m0_g0_s2(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = (np.maximum(0, x[majority, 1] + 10 * x[majority, 3] + 5 * x[majority, 6])) ** (1 / 2)
    m0[minority] = (np.maximum(0, 10 * x[minority, 1] + x[minority, 3] + 5 * x[minority, 6])) ** (1 / 2)

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = (np.maximum(0, x[majority, 0] + 10 * x[majority, 2] + 5 * x[majority, 5])) ** (1 / 2)
    g0[minority] = (np.maximum(0, 10 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5])) ** (1 / 2)

    return m0, g0


def m0_g0_s3(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = x[majority, 1] + 100 * x[majority, 3] + 5 * x[majority, 6]
    m0[minority] = 100 * x[minority, 1] + x[minority, 3] + 5 * x[minority, 6]

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = (np.maximum(0, x[majority, 0] + 100 * x[majority, 2] + 5 * x[majority, 5])) ** (1 / 2)
    g0[minority] = (np.maximum(0, 100 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5])) ** (1 / 2)

    return m0, g0


def m0_g0_s4(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # w_majority = 1 / np.linspace(1.0, 2.0, num=x.shape[1])
    # w_minority = 1 / np.logspace(1.0, 2.0, num=x.shape[1])
    #
    # m0 = np.zeros(x.shape[0], dtype=np.float64)
    # m0[majority] = gaussian(x[majority] @ w_majority)
    # m0[minority] = np.square(x[minority] @ w_minority)
    #
    # g0 = np.zeros(x.shape[0], dtype=np.float64)
    # g0[majority] = (np.maximum(0, x[majority, 0] + 100 * x[majority, 2] + 5 * x[majority, 5])) ** (1 / 2)
    # g0[minority] = (np.maximum(0, 100 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5])) ** (1 / 2)

    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = (np.maximum(0, x[majority, 0] + 100 * x[majority, 2] + 5 * x[majority, 6])) ** (1 / 2)
    m0[minority] = (np.maximum(0, 100 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 6])) ** (1 / 2)

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = (np.maximum(0, x[majority, 0] + 100 * x[majority, 2] + 5 * x[majority, 5])) ** (1 / 2)
    g0[minority] = (np.maximum(0, 100 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5])) ** (1 / 2)

    return m0, g0


def m0_g0_s5(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    # g0[majority] = x[majority, -1] + np.absolute(x[majority, 2]) + 0.5 * np.exp(x[majority, 5] + x[majority, 4])
    # g0[minority] = x[minority, -1] + np.absolute(-1.0 * x[minority, 3]) - 2.5 * x[minority, 4]

    g0 = x[:, -1] + np.absolute(x[:, 2]) + 0.5 * np.exp(x[:, 5] + x[:, 4])

    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = np.maximum(0, 0.5 * x[majority, 1] ** 2 + x[majority, 5] + x[majority, 3] ** 3)
    m0[minority] = np.maximum(0, -2.5 * x[minority, 1] ** 2 + x[minority, -1] + x[minority, 4])

    return m0, g0


def m0_g0_s6(x: np.ndarray, majority: np.ndarray, minority: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = (np.maximum(0, x[majority, 0] + 100 * x[majority, 2] + 5 * x[majority, 6])) ** (1 / 2)
    m0[minority] = (np.maximum(0, 100 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 6])) ** (1 / 2)

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = (np.maximum(0, x[majority, 1] + 100 * x[majority, 3] + 5 * x[majority, 5])) ** (1 / 2)
    g0[minority] = (np.maximum(0, 100 * x[minority, 1] + x[minority, 3] + 5 * x[minority, 5])) ** (1 / 2)

    return m0, g0


M0_G0_SETUPS = {
    "s1": m0_g0_s1,
    "s2": m0_g0_s2,
    "s3": m0_g0_s3,
    "s4": m0_g0_s4,
    "s5": m0_g0_s5,
    "s6": m0_g0_s6,
}


def m0_g0(x: np.ndarray, majority_s: float = 0.75, setup: str = "s1"):
    n_obs = x.shape[0]
    z = np.argsort(x[:, 0])
    threshold_idx = z[int(n_obs * majority_s)]
    threshold_val = x[threshold_idx, 0]
    majority = x[:, 0] < threshold_val
    minority = x[:, 0] >= threshold_val
    m0, g0 = M0_G0_SETUPS[setup](x, majority, minority)
    return m0, g0, (majority, minority)


class Data:

    def __train_test__(self, x, d, y, train_size: float = 0.5, seed: int = None, **kwargs):
        x_train, x_test, d_train, d_test, y_train, y_test = train_test_split(
            x, d, y, train_size=train_size, random_state=seed, shuffle=True
        )

        train = {"x": x_train, "d": d_train, "y": y_train}
        test = {"x": x_test, "d": d_test, "y": y_test}

        return train, test


class DataSynthetic(Data):
    def __ar_covariance_params__(self, dim: int, ar_rho: float):
        mu = np.zeros(dim,)

        rho = np.ones(dim,) * ar_rho
        sigma = np.zeros(shape=(dim, dim))
        for i in range(dim):
            for j in range(i, dim):
                sigma[i][j] = reduce(mul, [rho[k] for k in range(i, j)], 1)
        sigma = np.triu(sigma) + np.triu(sigma).T - np.diag(np.diag(sigma))

        return mu, sigma

    def __init__(
        self,
        nb_features: int = 9,
        nb_observations: int = 90,
        ar_rho: float = 0.8,
        sigma_v: float = 1.0,
        sigma_u: float = 1.0,
        majority_s: float = 0.75,
        m0_g0_setup: str = "s1",
        as_tensors: bool = True,
    ):
        super().__init__()

        self._nb_observations = nb_observations
        self._sigma_v = sigma_v
        self._sigma_u = sigma_u
        self._majority_s = majority_s
        self._m0_g0_setup = m0_g0_setup
        self._as_tensors = as_tensors
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mu, self.sigma = self.__ar_covariance_params__(dim=nb_features, ar_rho=ar_rho)

    @classmethod
    def init_from_opts(cls, opts, as_tensors: bool = True):
        return cls(
            nb_features=opts.nb_features,
            nb_observations=opts.nb_observations,
            ar_rho=opts.ar_rho,
            sigma_v=opts.sigma_v,
            sigma_u=opts.sigma_u,
            majority_s=opts.majority_s,
            m0_g0_setup=opts.m0_g0_setup,
            as_tensors=as_tensors,
        )

    @property
    def nb_observations(self):
        return self._nb_observations

    @nb_observations.setter
    def nb_observations(self, new_nb_observations: int):
        self._nb_observations = new_nb_observations

    @property
    def sigma_v(self):
        return self._sigma_v

    @sigma_v.setter
    def sigma_v(self, new_sigma_v: float):
        self._sigma_v = new_sigma_v

    @property
    def sigma_u(self):
        return self._sigma_u

    @sigma_u.setter
    def sigma_u(self, new_sigma_u: float):
        self._sigma_u = new_sigma_u

    @property
    def majority_s(self):
        return self._majority_s

    @majority_s.setter
    def majority_s(self, new_majority_s: float):
        self._majority_s = new_majority_s

    @property
    def m0_g0_setup(self):
        return self._m0_g0_setup

    @m0_g0_setup.setter
    def m0_g0_setup(self, new_m0_g0_setup: str):
        self._m0_g0_setup = new_m0_g0_setup

    @property
    def as_tensors(self):
        return self._as_tensors

    @as_tensors.setter
    def as_tensors(self, new_as_tensors: bool):
        self._as_tensors = new_as_tensors

    def prep(self, real_theta: float):
        x = self.rng.multivariate_normal(self.mu, self.sigma, self.nb_observations)
        m_0, g_0, (majority, minority) = m0_g0(x, majority_s=self.majority_s, setup=self.m0_g0_setup)

        d = m_0 + self.sigma_v * self.rng.randn(self.nb_observations)
        y = d * real_theta + g_0 + self.sigma_u * self.rng.randn(self.nb_observations,)

        if self.as_tensors:
            x = torch.tensor(x).to(self._device)
            d = torch.tensor(d).to(self._device)
            y = torch.tensor(y).to(self._device)
            majority = torch.tensor(majority).to(self._device)
            minority = torch.tensor(minority).to(self._device)

        return x, d, y, (majority, minority)

    def generate(self, real_theta: float, train_size: float = 0.5, seed: int = None, **kwargs):
        #####################################################################
        np.random.seed(seed)
        self.rng = np.random if seed is None else np.random.RandomState(seed)
        #####################################################################

        x, d, y, (majority, minority) = self.prep(real_theta)
        return self.train_test(x, d, y, majority, minority, train_size, seed)

    def train_test(self, x, d, y, majority, minority, train_size: float = 0.5, seed: int = None, **kwargs):
        (
            x_train,
            x_test,
            d_train,
            d_test,
            y_train,
            y_test,
            majority_train,
            majority_test,
            minority_train,
            minority_test,
        ) = train_test_split(x, d, y, majority, minority, train_size=train_size, random_state=seed, shuffle=True)

        train = {"x": x_train, "d": d_train, "y": y_train, "majority": majority_train, "minority": minority_train}
        test = {"x": x_test, "d": d_test, "y": y_test, "majority": majority_test, "minority": minority_test}

        return train, test
