import argparse
import numpy as np
import pandas as pd
import torch

from multiprocessing import Pool
from src.data.synthetic import DataSynthetic
from src.models.dml_pytorch import DoubleMachineLearningPyTorch
from src.models.sync_dml import SYNChronizedDoubleMachineLearning
from typing import Dict, Tuple
from tqdm import tqdm


def dml_theta_estimator(y: torch.Tensor, d: torch.Tensor, m_hat: torch.Tensor, l_hat: torch.Tensor) -> float:
    v_hat = d - m_hat
    u_hat = y - l_hat
    theta_hat = torch.mean(v_hat * u_hat) / torch.mean(v_hat * v_hat)
    return theta_hat.item()


def run_dml(opts, D1, D2, D2_a, D2_b, exp) -> Tuple[Dict, torch.Tensor, Dict]:
    preds = dict(theta=opts.real_theta, gamma=np.nan, exp=exp, method="DML")

    # Train on D1
    double_ml = DoubleMachineLearningPyTorch.init_from_opts(opts=opts)
    params = double_ml.fit_params(opts=opts)
    double_ml = double_ml.fit(train=D1, test=D2, **params)

    # Predict theta on D2_a
    m_hat, l_hat = double_ml.predict(D2_a["x"])
    theta_init = dml_theta_estimator(y=D2_a["y"], d=D2_a["d"], m_hat=m_hat, l_hat=l_hat)
    u_hat, v_hat = D2_a["y"] - l_hat, D2_a["d"] - m_hat
    stats = {
        "corr.abs": torch.mean(torch.absolute(u_hat * v_hat)).item(),
        "res_m.2": torch.mean(v_hat ** 2).item(),
        "res_l.2": torch.mean(u_hat ** 2).item()
    }

    # Predict g(X) on D2_b
    m_hat, l_hat = double_ml.predict(D2_b["x"])
    g_hat = l_hat - theta_init * m_hat

    # Predict final theta on D2
    m_hat, l_hat = double_ml.predict(D2["x"])
    dml_theta = dml_theta_estimator(y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)
    preds["theta_hat"] = dml_theta
    preds["bias"] = dml_theta - opts.real_theta

    return preds, g_hat, stats


def run_cdml(opts: argparse.Namespace, D1, D2, D2_a, D2_b, exp, g_hat, dml_stats) -> Dict:
    results = []
    for gamma in opts.gammas:
        opts.sync_dml_start_gamma = gamma

        sync_dml = SYNChronizedDoubleMachineLearning.init_from_opts(opts=opts)
        params = sync_dml.fit_params(opts=opts, dml_stats=dml_stats)

        preds = dict(theta=opts.real_theta, gamma=gamma, exp=exp, method="C-DML")

        # Train on D1
        sync_dml = sync_dml.fit(train=D1, test=D2, **params)

        # Predict theta on D2_a
        m_hat, l_hat = sync_dml.predict(D2_a["x"])
        sync_dml_theta_for_cv = dml_theta_estimator(y=D2_a["y"], d=D2_a["d"], m_hat=m_hat, l_hat=l_hat)

        # Residual error on D2_b
        sync_dml_y_hat = g_hat + sync_dml_theta_for_cv * D2_b["d"]
        preds["y_res.2"] = torch.mean((D2_b["y"] - sync_dml_y_hat) ** 2).item()

        # Predict final theta on D2
        m_hat, l_hat = sync_dml.predict(D2["x"])
        sync_dml_theta = dml_theta_estimator(y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)
        preds["theta_hat"] = sync_dml_theta
        preds["bias"] = sync_dml_theta - opts.real_theta

        results.append(preds)

    results = pd.DataFrame(results)
    results = results.sort_values(by="y_res.2", ascending=True)
    preds = results.iloc[0].squeeze().to_dict()

    return preds


def run_experiment(opts: argparse.Namespace, exp: int) -> Dict[str, Dict]:
    data = DataSynthetic.init_from_opts(opts, as_tensors=True)

    D1, D2 = data.generate(real_theta=opts.real_theta, train_size=0.5, seed=exp)
    D2_copy = {k: v.detach().clone() for k, v in D2.items()}
    D2_a, D2_b = data.__train_test__(x=D2_copy["x"], d=D2_copy["d"], y=D2_copy["y"], train_size=0.5, seed=exp)

    dml_results, g_hat, dml_stats = run_dml(opts, D1, D2, D2_a, D2_b, exp)
    cdml_results = run_cdml(opts, D1, D2, D2_a, D2_b, exp, g_hat, dml_stats)

    return {
        "dml_results": dml_results,
        "cdml_results": cdml_results,
    }


def run_cv(opts: argparse.Namespace) -> pd.DataFrame:

    pbar = tqdm(total=opts.n_exp, desc=f"running C-DML")

    def _update(*a):
        pbar.update()

    with Pool(processes=opts.n_processes) as pool:
        tasks = [
            pool.apply_async(run_experiment, args=(opts, i), callback=_update)
            for i in range(opts.n_exp)
        ]
        [task.wait() for task in tasks]
        results = [task.get() for task in tasks]

    pbar.close()

    results = [result["dml_results"] for result in results] + [result["cdml_results"] for result in results]
    return pd.DataFrame(results)
