import json
import numpy as np
import pandas as pd
import random
import string

from argparse import Namespace
from pathlib import Path


def gen_random_string(length: int):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def dump_opts_to_json(opts: Namespace, path: Path):
    params = vars(opts)
    with open(path, "w") as f:
        json.dump(params, f, indent=4)


def calc_cv_optimal_gamma_for_experiment(df: pd.DataFrame, thresh: float, sort_by: str = "y_res.2") -> float:
    df_x = df[~np.isnan(df["gamma"])]
    ref_df_sorted = df_x.sort_values(by=sort_by, ascending=True)
    optimal_gamma = ref_df_sorted.iloc[0]["gamma"].item()

    return optimal_gamma
