"""
"""

import os

import matplotlib.cm as cm
import numpy as np
from matplotlib import lines as mlines
from matplotlib import pyplot as plt

from dsps.metallicity import umzr

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def make_mzr_comparison_plot(
    params,
    params2=umzr.DEFAULT_MZR_PARAMS,
    fname=None,
    label1=r"${\rm new\ model}$",
    label2=r"${\rm default\ model}$",
):
    """Make basic diagnostic plot of the model for Tburst

    Parameters
    ----------
    params : namedtuple
        Instance of umzr.MZRParams

    params2 : namedtuple, optional
        Instance of umzr.MZRParams
        Default is set by DEFAULT_MZR_PARAMS

    fname : string, optional
        filename of the output figure

    """

    n_t = 5
    colors = cm.coolwarm(np.linspace(1, 0, n_t))  # red first
    t_arr = np.linspace(2, 14, n_t)

    n_sm = 1_000
    logsm_arr = np.linspace(5, 12, n_sm)

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(-6.5, -1.1)
    ax.set_xlim(5.1, 11.9)

    for it in range(n_t):
        ax.plot(
            logsm_arr, umzr.mzr_model(logsm_arr, t_arr[it], *params), color=colors[it]
        )
        ax.plot(
            logsm_arr,
            umzr.mzr_model(logsm_arr, t_arr[it], *params2),
            "--",
            color=colors[it],
        )

    drn = os.path.join(
        os.path.join(os.path.dirname(_THIS_DRNAME), "tests"), "testing_data"
    )
    bn = "maiolino_etal18_mzr.dat"
    fn = os.path.join(drn, bn)
    m18_data = np.loadtxt(fn, delimiter=",")
    lgsm_msun_m18 = m18_data[:, 0]
    lgz_zsun_m18 = m18_data[:, 1]
    lgzmet_m18 = lgz_zsun_m18 + umzr.LGMET_SOLAR
    ax.scatter(lgsm_msun_m18[::2], lgzmet_m18[::2], s=100, color="k", marker="*")

    red_line = mlines.Line2D([], [], ls="-", c=colors[0], label=r"${\rm z=3}$")
    blue_line = mlines.Line2D([], [], ls="-", c=colors[-1], label=r"${\rm z=0}$")
    solid_line = mlines.Line2D([], [], ls="-", c="gray", label=label2)
    dashed_line = mlines.Line2D([], [], ls="--", c="gray", label=label1)
    black_star = mlines.Line2D(
        [],
        [],
        color="k",
        marker="*",
        linestyle="None",
        markersize=10,
        label=r"${\rm Maiolino}$+$18\ (z=0)$",
    )

    leg0 = ax.legend(handles=[blue_line, red_line], loc="upper left")
    ax.add_artist(leg0)
    ax.legend(handles=[black_star, solid_line, dashed_line], loc="lower right")
    xlabel = ax.set_xlabel(r"$\log_{10}M_{\star}/M_{\odot}$")
    ylabel = ax.set_ylabel(r"$\log_{10}Z$")
    ax.set_title(r"${\rm mass}$--${\rm metallicity\ relation}$")

    if fname is not None:
        fig.savefig(
            fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
        )
    return fig
