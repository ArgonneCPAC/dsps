"""
"""
from collections import OrderedDict
import os
import h5py
import numpy as np
from jax import vmap, jit as jjit
from jax import numpy as jnp
from .sfh_model import DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS, DEFAULT_MAH_PARAMS
from .mzr import DEFAULT_MZR_PARAMS


try:
    from diffsfh.differential_sfr_floor import (
        _get_bounded_params as _get_bounded_ms_params,
    )
    from diffsfh.galaxy_quenching import (
        _get_bounded_params_rejuv as _get_bounded_q_params,
    )

    HAS_DIFFSFH = True
except ImportError:
    HAS_DIFFSFH = False

TASSO_DRN = "/Users/aphearin/work/DATA/diffstar_data"
BEBOP_DRN = "/lcrc/project/halotools/diffstar_data"
DIFFSFH_BN = "run2_bpl_small_diffsfh_rejuv_depl_floor_fixed_k_hi_lagprior_minmass7.0"
DIFFMAH_BN = "run2_bpl_small_diffmah"
UMACHINE_BN = "um_histories_subsample_dr1_bpl_cens_diffmah.npy"


MAH_FIT_KEYS = ("logt0", "logmpeak", "mah_logtc", "mah_k", "early_index", "late_index")
MS_FIT_KEYS = [
    "lgmcrit",
    "lgy_at_mcrit",
    "indx_k",
    "indx_lo",
    "indx_hi",
    "floor_low",
    "tau_dep",
]
Q_FIT_KEYS = ["qt", "qs", "q_drop", "q_rejuv"]


@jjit
def get_bounded_ms_p(u_ms_params):
    return _get_bounded_ms_params(*u_ms_params)


@jjit
def get_bounded_q_p(u_q_params):
    return _get_bounded_q_params(*u_q_params)


_a = [0]
get_bounded_ms_p_vmap = jjit(vmap(get_bounded_ms_p, in_axes=_a))
get_bounded_q_p_vmap = jjit(vmap(get_bounded_q_p, in_axes=_a))


def load_small_bpl_fits(
    umachine_bn=UMACHINE_BN,
    diffmah_bn=DIFFMAH_BN,
    diffsfh_bn=DIFFSFH_BN,
    diffstar_drn=TASSO_DRN,
):
    if not HAS_DIFFSFH:
        raise ImportError("Must have diffsfh installed to use load_small_bpl_fits")
    umachine = np.load(os.path.join(diffstar_drn, umachine_bn))

    t_bpl = np.load(os.path.join(diffstar_drn, "bpl_cosmic_time.npy"))
    a_bpl = np.loadtxt(os.path.join(diffstar_drn, "scale_list_bpl.dat"))
    z_bpl = 1 / a_bpl - 1

    diffmah_params = OrderedDict()
    with h5py.File(os.path.join(diffstar_drn, diffmah_bn), "r") as hdf:
        for key in hdf.keys():
            diffmah_params[key] = hdf[key][...]

    diffstar_params = OrderedDict()
    with h5py.File(os.path.join(diffstar_drn, diffsfh_bn), "r") as hdf:
        for key in hdf.keys():
            diffstar_params[key] = hdf[key][...]

    assert np.allclose(diffmah_params["halo_id"], umachine["halo_id"])
    diffmah_params.pop("halo_id")
    assert np.allclose(diffstar_params["halo_id"], umachine["halo_id"])
    diffstar_params.pop("halo_id")

    mock = OrderedDict([(key, umachine[key]) for key in umachine.dtype.names])
    mock.update(diffmah_params)
    mock.update(diffstar_params)
    mock["logt0"] = np.log10(mock.pop("t0"))
    n_gal = mock["logt0"].size

    for fit_key, new_key in zip(MAH_FIT_KEYS, DEFAULT_MAH_PARAMS.keys()):
        mock[new_key] = mock.pop(fit_key)
    for key in MS_FIT_KEYS:
        mock["u_" + key] = mock.pop(key)
    for key in Q_FIT_KEYS:
        mock["u_" + key] = mock.pop(key)

    u_ms_params = jnp.array([mock["u_" + key] for key in MS_FIT_KEYS]).T
    u_q_params = jnp.array([mock["u_" + key] for key in Q_FIT_KEYS]).T

    ms_params = get_bounded_ms_p_vmap(u_ms_params)
    q_params = get_bounded_q_p_vmap(u_q_params)

    for key, arr in zip(DEFAULT_MS_PARAMS.keys(), ms_params):
        mock[key] = arr
    for key, arr in zip(DEFAULT_Q_PARAMS.keys(), q_params):
        mock[key] = arr

    for key, val in DEFAULT_MZR_PARAMS.items():
        mock[key] = np.zeros(n_gal) + val

    return mock, t_bpl, z_bpl


def get_random_q_params(
    n,
    lg_qt=DEFAULT_Q_PARAMS["lg_qt"],
    lg_qs=DEFAULT_Q_PARAMS["lg_qs"],
    lg_drop=DEFAULT_Q_PARAMS["lg_drop"],
    lg_rejuv=DEFAULT_Q_PARAMS["lg_rejuv"],
    lg_qt_scatter=0.01,
    lg_qs_scatter=0.01,
    lg_drop_scatter=0.01,
    lg_rejuv_scatter=0.01,
):
    zz = np.zeros(n)
    lg_qt_sample = np.random.normal(loc=lg_qt + zz, scale=lg_qt_scatter)
    lg_qs_sample = np.random.normal(loc=lg_qs + zz, scale=lg_qs_scatter)
    lg_drop_sample = np.random.normal(loc=lg_drop + zz, scale=lg_drop_scatter)
    lg_rejuv_sample = np.random.normal(loc=lg_rejuv + zz, scale=lg_rejuv_scatter)
    return np.vstack((lg_qt_sample, lg_qs_sample, lg_drop_sample, lg_rejuv_sample)).T
