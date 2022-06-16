import argparse
import multiprocessing as mp
import numpy as np
import torch

from src.data import DATA_TYPES
from src.modules.gamma_scheduler import GAMMA_SCHEDULERS
from src.modules.loss import LOSSES
from src.modules.nn_nets import Nets
from pathlib import Path
from src.utils.general import gen_random_string


def add_common_args(parser) -> argparse.ArgumentParser:
    """
    Parse shared args.
    :param parser: argparser object.
    :return: argparser object.
    """
    test = parser.add_argument_group("Test Parameters")
    test.add_argument("--n-processes", type=int, default=(mp.cpu_count() - 1) // 2, help="number of processes to launch")
    test.add_argument("--n-exp", type=int, default=500, help="number of experiments to run")
    test.add_argument("--seed", type=int, default=None, help="random seed")
    test.add_argument("--output-dir", type=Path, default=Path("results"))
    test.add_argument("--name", type=str, default=gen_random_string(5), help="experiment name")
    test.add_argument("--real-theta", type=str, default="0.0", help="true value of theta")

    return parser


def add_data_args(parser) -> argparse.ArgumentParser:
    """
    Parse shared data args.
    :param parser: argparser object.
    :return: argparser object.
    """
    parser.add_argument("--data-type", type=str, default="synthetic", choices=DATA_TYPES.keys())
    parser.add_argument("--nb-features", type=int, default=10, help="number of high-dimensional features")
    parser.add_argument("--nb-observations", type=int, default=2000, help="number of observations")
    parser.add_argument("--sigma-v", type=float, default=1.0, help="V ~ N(0,sigma)")
    parser.add_argument("--sigma-u", type=float, default=1.0, help="U ~ N(0,sigma)")

    syn_data = parser.add_argument_group("Synthetic Data Parameters")
    syn_data.add_argument("--ar-rho", type=float, default=0.8, help="AutoRegressive(rho) coefficient")
    syn_data.add_argument("--majority-s", type=float, default=0.75, help="majority split value")
    syn_data.add_argument("--m0-g0-setup", type=str, default="s1", choices=("s1", "s2", "s3", "s4", "s5"))

    return parser


def add_double_ml_args(parser) -> argparse.ArgumentParser:
    dml = parser.add_argument_group("Double Machine Learning Parameters")
    dml.add_argument("--dml-net", type=str, default="NonLinearNet", choices=Nets.keys())
    dml.add_argument("--dml-lr", type=float, default=0.01, help="learning rate")
    dml.add_argument("--dml-clip-grad-norm", type=float, default=3.0)
    dml.add_argument("--dml-epochs", type=int, default=2000)
    dml.add_argument("--dml-optimizer", type=str, default="SGD", help="torch.optim.Optimizer name")
    return parser


def add_sync_dml_args(parser) -> argparse.ArgumentParser:
    sync_dml = parser.add_argument_group("SYNChronized (Double) Machine Learning Parameters")
    sync_dml.add_argument("--sync-dml-warmup-with-dml", action="store_true", default=False)
    sync_dml.add_argument("--sync-dml-net", type=str, default="NonLinearNet", choices=Nets.keys())
    sync_dml.add_argument("--sync-dml-loss", type=str, default="CorrelationLoss", choices=LOSSES.keys())
    sync_dml.add_argument("--sync-dml-lr", type=float, default=0.01, help="learning rate")
    sync_dml.add_argument("--sync-dml-clip-grad-norm", type=float, default=3.0)
    sync_dml.add_argument("--sync-dml-epochs", type=int, default=2000)
    sync_dml.add_argument("--sync-dml-optimizer", type=str, default="SGD", help="torch.optim.Optimizer name")

    sync_dml.add_argument("--sync-dml-gamma-scheduler", type=str, default="FixedGamma", choices=GAMMA_SCHEDULERS.keys())
    sync_dml.add_argument("--sync-dml-warmup-epochs", type=int, default=1000)
    sync_dml.add_argument("--sync-dml-start-gamma", type=float, default=1.0)
    sync_dml.add_argument("--sync-dml-end-gamma", type=float, default=1.0)

    return parser


def set_seed(seed: int):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


class Parser(object):
    @staticmethod
    def double_ml() -> argparse.Namespace:
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()
        parser = add_common_args(parser)
        parser = add_data_args(parser)
        parser = add_double_ml_args(parser)
        opt = parser.parse_args(args=[])

        set_seed(opt.seed)
        opt.output_dir = opt.output_dir.expanduser()
        opt.output_dir.mkdir(parents=True, exist_ok=True)
        opt.real_theta = float(opt.real_theta)

        return opt

    @staticmethod
    def sync_dml() -> argparse.Namespace:
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()
        parser = add_common_args(parser)
        parser = add_data_args(parser)
        parser = add_sync_dml_args(parser)
        opt = parser.parse_args(args=[])

        set_seed(opt.seed)
        opt.output_dir = opt.output_dir.expanduser()
        opt.output_dir.mkdir(parents=True, exist_ok=True)
        opt.real_theta = float(opt.real_theta)

        return opt

    @staticmethod
    def compare() -> argparse.Namespace:
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()
        parser = add_common_args(parser)
        parser = add_data_args(parser)
        parser = add_double_ml_args(parser)
        parser = add_sync_dml_args(parser)

        regression = parser.add_argument_group("Regression Parameters")
        regression.add_argument(
            "--thetas", type=str, nargs="+", default=[" 0.0", " 10.0"],
        )
        regression.add_argument(
            "--gammas",
            type=str,
            nargs="+",
            default=[
                " 0.000",
                " 0.001",
                " 0.01",
                " 0.1",
                " 1.0",
                " 10.",
                " 100.",
                " 1000.",
            ],
        )
        opt = parser.parse_args(args=[])

        set_seed(opt.seed)
        opt.output_dir = opt.output_dir.expanduser()
        opt.output_dir.mkdir(parents=True, exist_ok=True)
        opt.thetas = [float(theta) for theta in opt.thetas]
        opt.n_thetas = len(opt.thetas)
        opt.gammas = [float(gamma) for gamma in opt.gammas]
        opt.n_gammas = len(opt.gammas)

        return opt
