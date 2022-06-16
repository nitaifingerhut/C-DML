import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams["font.family"] = "serif"

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import Any, Dict, Union


COLORS = (
    ("dimgray", "lightgray"),
    ("royalblue", "deepskyblue"),
    ("mediumvioletred", "palevioletred"),
    ("yellowgreen", "olivedrab"),
)
FACECOLORS = ("C0", "C2", "C4", "C6")


def plot_method_statistics(df: pd.DataFrame, n_exp: float, save_as: pathlib.Path):
    nb_columns = len(df.columns)
    _, axs = plt.subplots(1, nb_columns, figsize=(5 * nb_columns, 5))

    curr_axs = 0
    if "theta estimation" in df.columns:
        mean_theta_estimation = np.mean(df["theta estimation"].to_numpy())
        sns.histplot(data=df["theta estimation"], alpha=0.5, ax=axs[curr_axs], bins=20, kde=True)
        axs[curr_axs].set_title("$\hat{\\theta}$" + "\n" + "MEAN = {:.3f}".format(mean_theta_estimation))
        curr_axs += 1

    if "empirical bias" in df.columns:
        empirical_bias = df["empirical bias"].to_numpy()
        mean_bias = np.mean(empirical_bias)
        stderr = np.std(empirical_bias) / n_exp

        p = sns.violinplot(data=df["empirical bias"], ax=axs[curr_axs], orient="v")
        p.set_title(
            "empirical bias" + "($\\Delta \\theta$)" + "\n" + "MEAN = {:.3f} ± {:.3f}".format(mean_bias, stderr)
        )
        p.set_xlabel("")
        p.set_ylabel("")
        axs[curr_axs].axhline(y=0, linestyle="--", alpha=0.5, color="red")
        curr_axs += 1

    if "residuals correlation" in df.columns:
        mean_residuals_correlation = np.mean(df["residuals correlation"].to_numpy())

        p = sns.boxplot(data=df["residuals correlation"], ax=axs[curr_axs], orient="v")
        p.set_title("residuals correlation" + "\n" + "MEAN = {:.3f}".format(mean_residuals_correlation))
        p.set_xlabel("")
        p.set_ylabel("")
        hlines = set([round(i) for i in axs[curr_axs].get_ylim()])
        for hline in hlines:
            axs[curr_axs].axhline(y=hline, linestyle="--", alpha=0.5, color="red")
        curr_axs += 1

    if "res_m.2" in df.columns:
        mean_res_m_2 = np.mean(df["res_m.2"].to_numpy())

        p = sns.boxplot(data=df["res_m.2"], ax=axs[curr_axs], orient="v")
        p.set_title(r"$\left( D - \hat{m}_0 \right)^2$" + "\n" + "MEAN = {:.3e}".format(mean_res_m_2))
        p.set_xlabel("")
        p.set_ylabel("")
        p.set_ylim(bottom=0.0, top=np.percentile(df["res_m.2"].to_numpy(), 95))
        p.set_yscale("symlog", linthresh=np.percentile(df["res_m.2"].to_numpy(), 50))

        curr_axs += 1

    if "res_l.2" in df.columns:
        mean_res_l_2 = np.mean(df["res_l.2"].to_numpy())

        p = sns.boxplot(data=df["res_l.2"], ax=axs[curr_axs], orient="v")
        p.set_title(r"$\left( Y - \hat{\ell}_0 \right)^2$" + "\n" + "MEAN = {:.3e}".format(mean_res_l_2))
        p.set_xlabel("")
        p.set_ylabel("")
        p.set_ylim(bottom=0.0, top=np.percentile(df["res_l.2"].to_numpy(), 95))
        p.set_yscale("symlog", linthresh=np.percentile(df["res_l.2"].to_numpy(), 50))

        curr_axs += 1

    if "delta.m.2" in df.columns:
        mean_delta_m_2 = np.mean(df["delta.m.2"].to_numpy())

        p = sns.boxplot(data=df["delta.m.2"], ax=axs[curr_axs], orient="v")
        p.set_title("$(m_0(X) - \hat{m}_0) ^ 2$" + "\n" + "MEAN = {:.3e}".format(mean_delta_m_2))
        p.set_xlabel("")
        p.set_ylabel("")
        p.set_ylim(bottom=0.0, top=np.percentile(df["delta.m.2"].to_numpy(), 95))
        p.set_yscale("symlog", linthresh=np.percentile(df["delta.m.2"].to_numpy(), 50))

        curr_axs += 1

    if "delta.g.2" in df.columns:
        mean_delta_g_2 = np.mean(df["delta.g.2"].to_numpy())

        p = sns.boxplot(data=df["delta.g.2"], ax=axs[curr_axs], orient="v")
        p.set_title("$(g_0(X) - \hat{g}_0) ^ 2$" + "\n" + "MEAN = {:.3e}".format(mean_delta_g_2))
        p.set_xlabel("")
        p.set_ylabel("")
        p.set_ylim(bottom=0.0, top=np.percentile(df["delta.g.2"].to_numpy(), 95))
        p.set_yscale("symlog", linthresh=np.percentile(df["delta.g.2"].to_numpy(), 50))

        curr_axs += 1

    assert curr_axs == nb_columns

    plt.subplots_adjust(wspace=0.25, hspace=0.4)
    plt.savefig(save_as, bbox_inches="tight")
    plt.close()


def plot_model_statistics(history: Dict[str, Any], theta: float, save_as: pathlib.Path):

    required_attr = set(["train loss", "train theta-est", "test loss", "test theta-est"])
    intersection = set(history.keys()).intersection(required_attr)
    if intersection != required_attr:
        raise RuntimeError

    batches = np.arange(start=0, stop=len(history["train-loss"]))
    epochs = np.arange(start=0, stop=len(history["test-loss"]))

    best_test = np.asarray(history["test-loss"]).argmin()
    best_theta_est = history["test-theta-est"][best_test]

    _, axs = plt.subplots(2, 2, figsize=(7.5, 7.5))

    axs[0, 0].plot(batches, history["train-loss"])
    axs[0, 0].set_ylabel("loss")
    axs[0, 0].set_xlabel("batch")
    axs[0, 0].set_title("train loss")
    axs[0, 0].set_yscale("log")

    axs[0, 1].plot(batches, history["train-theta-est"])
    axs[0, 1].set_ylabel("theta estimation")
    axs[0, 1].set_xlabel("batch")
    axs[0, 1].set_title("train theta estimation")
    axs[0, 1].axhline(y=theta, color="red", alpha=0.8, linestyle="--")

    axs[1, 0].plot(epochs, history["test-loss"])
    axs[1, 0].set_ylabel("loss")
    axs[1, 0].set_xlabel("epoch")
    axs[1, 0].set_title("test loss")
    axs[1, 0].set_yscale("log")
    axs[1, 0].axvline(x=best_test, color="red", alpha=0.8, linestyle="--")

    axs[1, 1].plot(batches, history["test-theta-est"])
    axs[1, 1].set_ylabel("test estimation")
    axs[1, 1].set_xlabel("epoch")
    axs[1, 1].set_title("test theta estimation (best= {:.3f})".format(best_theta_est))
    axs[1, 1].axvline(x=best_test, color="red", alpha=0.8, linestyle="--")
    axs[1, 1].axhline(y=theta, color="red", alpha=0.8, linestyle="--")

    plt.suptitle("$\\theta_0$ = " + f"{theta}")

    plt.subplots_adjust(wspace=0.15, hspace=0.50)
    plt.savefig(save_as, bbox_inches="tight")
    plt.close()


def create_axis_grid(num_plots: int, sz: float = 5.0):

    x = math.sqrt(num_plots)
    rows = round(x)
    cols = math.ceil(x)

    fig_height = sz * cols / rows
    fig_width = sz * rows / cols

    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if rows == 1:
        axs = np.array(axs).reshape((1, cols))

    for r, c in zip(range(rows), range(cols)):
        axs[r, c].set_xticks([])
        axs[r, c].set_yticks([])

    # divider = make_axes_locatable(axs.ravel())
    # cax = divider.append_axes('right', size='5%', pad=0.05)

    return axs, rows, cols


def boxplot_thetas_bias(
    df: pd.DataFrame, n_thetas: int, h_sz: float = 2.5, suptitle: str = "", save_as: Union[str, Path] = None,
):
    w_sz = n_thetas * h_sz

    plt.figure(figsize=(w_sz, h_sz))
    plot = sns.catplot(data=df, col="theta", x="gamma", y="bias", kind="violin", legend_out=True)
    axes = plot.axes.squeeze()
    for ax in axes:
        ax.axhline(y=0.0, color="red", alpha=0.5, linestyle="--")

        # Update title
        t = ax.get_title()
        theta = float(t.split(" = ")[1])
        ax.set_title("$\\theta_0$ = " + f"{theta}")

    # plt.suptitle(suptitle)
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()


def boxplot_cv_gammas(
    df: pd.DataFrame, n_thetas: int, h_sz: float = 2.5, suptitle: str = "", save_as: Union[str, Path] = None,
):

    w_sz = n_thetas * h_sz

    plt.figure(figsize=(w_sz, h_sz))
    plot = sns.catplot(data=df, col="theta", x="gamma", kind="count", legend_out=True)
    axes = plot.axes.squeeze()
    for ax in axes:
        # Update title
        t = ax.get_title()
        theta = float(t.split(" = ")[1])
        ax.set_title("$\\theta_0$ = " + f"{theta}")

    # plt.suptitle(suptitle)
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()


def lineplot_thetas_squared_bias(
    df: pd.DataFrame, y_key: str, y_label: str, suptitle: str = "", save_as: Union[str, Path] = None,
):

    plt.figure()
    x = sns.lineplot(data=df, x="theta", y=y_key, hue="gamma", style="gamma", markers=True, palette="Set1")

    plt.ylabel(y_label)
    plt.xlabel("$\\theta_0$")
    # plt.suptitle(suptitle)
    plt.legend(title="$\gamma$")
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()


def lineplot_thetas_log_squared_y_residual_error(
    df: pd.DataFrame, y_key: str, y_label: str, suptitle: str = "", save_as: Union[str, Path] = None,
):

    plt.figure()
    x = sns.lineplot(
        data=df, x="theta", y=y_key, hue="gamma", style="gamma", hue_norm=mplc.LogNorm(), markers=True, palette="Set1"
    )

    plt.ylabel(y_label)
    plt.xlabel("$\\theta_0$")
    # plt.suptitle(suptitle)
    plt.legend(title="$\gamma$")
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()


def boxplot_final_biases(
    df: pd.DataFrame, n_thetas: int, h_sz: float = 2.5, suptitle: str = "", save_as: Union[str, Path] = None,
):
    df = df.sort_values(by=["theta"])
    df_summary = df.groupby(["theta", "method"]).mean().reset_index()

    w_sz = n_thetas * h_sz

    plt.figure(figsize=(w_sz, h_sz))
    plot = sns.catplot(data=df, col="theta", x="method", y="bias", kind="violin", legend_out=True)
    axes = plot.axes.squeeze()
    for ax in axes:
        # Add horizontal line
        ax.axhline(y=0.0, color="red", alpha=0.5, linestyle="--")

        # Update title with mean values
        t = ax.get_title()
        theta = float(t.split(" = ")[1])
        theta_summary = df_summary[df_summary["theta"] == theta]
        dml_mean_bias = theta_summary[theta_summary["method"] == "DML"]["bias"].item()
        cv_mean_bias = theta_summary[theta_summary["method"] == "SYNC-ML"]["bias"].item()
        summary_str = "mean bias: DML = {:.4f} | SYNC-ML = {:.4f}".format(dml_mean_bias, cv_mean_bias)
        ax.set_title("$\\theta_0$ = " + f"{theta} \n" + summary_str)
        ax.set(xlabel=None)

    # plt.suptitle(suptitle)
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()
