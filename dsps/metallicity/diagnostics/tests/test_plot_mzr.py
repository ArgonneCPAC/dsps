"""
"""

import os

from ... import umzr
from .. import plot_mzr as pmzr

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_plot_mzr_function():
    fn = os.path.join(_THIS_DRNAME, "dummy.png")
    pmzr.make_mzr_comparison_plot(umzr.DEFAULT_MZR_PARAMS, fname=fn)
    assert os.path.isfile(fn)
    os.remove(fn)
