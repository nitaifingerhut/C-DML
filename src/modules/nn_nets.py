import inspect
import sys
import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.net_m = self.net_m.cuda()
            self.net_l = self.net_l.cuda()

        self.net_m = self.net_m.type(torch.float64)
        self.net_l = self.net_l.type(torch.float64)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        m_x = self.net_m(x).squeeze()
        l_x = self.net_l(x).squeeze()
        return m_x, l_x


class LinearNet(BaseNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net_m = nn.Linear(in_features=in_features, out_features=1)
        self.net_l = nn.Linear(in_features=in_features, out_features=1)

        self.__post_init__()


class NonLinearNet(BaseNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 2, out_features=in_features // 4),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 4, out_features=1),
        )

        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 2, out_features=in_features // 4),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 4, out_features=1),
        )

        self.__post_init__()


############################################################
class ExpressiveNet(BaseNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )

        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )

        self.__post_init__()


############################################################


class SharedNet(nn.Module):
    def __init__(self):
        super().__init__()

    def __post_init__(self):
        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.net = self.net.type(torch.float64)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pred = self.net(x).squeeze()
        m_x, l_x = pred[:, 0], pred[:, 1]
        return m_x, l_x


class SharedLinearNet(SharedNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net = nn.Linear(in_features=in_features, out_features=2)

        self.__post_init__()


class SharedNonLinearNet(SharedNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features=in_features // 2, out_features=2),
        )

        self.__post_init__()


Nets = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if obj.__module__ is __name__
}
