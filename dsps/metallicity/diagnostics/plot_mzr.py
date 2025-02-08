"""
"""

import os

import numpy as np

try:
    import matplotlib.cm as cm
    from matplotlib import lines as mlines
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .. import umzr

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))

MGREEN = "#2ca02c"
MPURPLE = "#9467bd"


def make_mzr_comparison_plot(
    params,
    params2=umzr.DEFAULT_MZR_PARAMS,
    fname=None,
    label1=r"${\rm new\ model}$",
    label2=r"${\rm default\ model}$",
):
    """Make basic diagnostic plot of mass-metallity model.

    Compare to Maiolino+18, arXiv:

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
    assert HAS_MATPLOTLIB, "Must have matplotlb installed to use this function"

    n_t = 5
    colors = cm.coolwarm(np.linspace(1, 0, n_t))  # red first
    t_arr = np.linspace(2, 14, n_t)

    n_sm = 1_000
    logsm_arr = np.linspace(2, 14, n_sm)

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(-7.5, -1.1)
    ax.set_xlim(4.1, 12.4)

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
    ax.plot(
        lgsm_msun_m18[::2], lgzmet_m18[::2], color="k", marker="*", linestyle="None"
    )

    bn = "kirby_etal13_mzr.dat"
    fn = os.path.join(drn, bn)
    k13_data = np.loadtxt(fn, delimiter=",")
    lgsm_msun_k13 = k13_data[:, 0]
    lgz_zsun_k13 = k13_data[:, 1]
    lgzmet_k13 = lgz_zsun_k13 + umzr.LGMET_SOLAR

    ax.plot(
        lgsm_msun_k13[::2],
        lgzmet_k13[::2],
        color=MGREEN,
        marker="v",
        linestyle="None",
    )

    bn = "galazzi_etal05_mzr.dat"
    fn = os.path.join(drn, bn)
    g05_data = np.loadtxt(fn, delimiter=",")
    lgsm_msun_g05 = g05_data[:, 0]
    lgz_zsun_g05 = g05_data[:, 1]
    lgzmet_g05 = lgz_zsun_g05 + umzr.LGMET_SOLAR

    ax.plot(
        lgsm_msun_g05[::2],
        lgzmet_g05[::2],
        color=MPURPLE,
        marker="d",
        linestyle="None",
    )

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
        markersize=6,
        label=r"${\rm Maiolino}$+$18\ (z=0)$",
    )
    green_triangle = mlines.Line2D(
        [],
        [],
        color=MGREEN,
        marker="v",
        linestyle="None",
        markersize=6,
        label=r"${\rm Kirby}$+$13\ (z=0)$",
    )

    purple_square = mlines.Line2D(
        [],
        [],
        color=MPURPLE,
        marker="d",
        linestyle="None",
        markersize=6,
        label=r"${\rm Gallazzi}$+$05\ (z=0)$",
    )

    leg0 = ax.legend(handles=[blue_line, red_line], loc="upper left")
    ax.add_artist(leg0)
    ax.legend(
        handles=[purple_square, black_star, green_triangle, solid_line, dashed_line],
        loc="lower right",
    )
    xlabel = ax.set_xlabel(r"$\log_{10}M_{\star}/M_{\odot}$")
    ylabel = ax.set_ylabel(r"$\log_{10}Z$")
    ax.set_title(r"${\rm mass}$--${\rm metallicity\ relation}$")

    if fname is not None:
        fig.savefig(
            fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
        )
    return fig
