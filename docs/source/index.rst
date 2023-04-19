DSPS: Differentiable Stellar Population Synthesis
=================================================

DSPS is a python library for stellar population synthesis (SPS) written with 
`JAX <https://jax.readthedocs.io/>`__.
You can use DSPS to calculate the SED and photometry of a galaxy as a function of its 
star formation history, metallicity, dust, and other properties.
Typical applications of DSPS include fitting the SED of an individual galaxy, 
and making predictions for the SEDs and colors of a galaxy population.

DSPS is open-source code that is publicly available on
`GitHub <https://github.com/ArgonneCPAC/dsps/>`__. 
You can find more information about DSPS in 
`our paper <https://arxiv.org/abs/2112.06830/>`__.
These docs show you how to use the core functionality of DSPS,
and also provide tutorial material on differentiable programming with JAX.

User Guide
----------

.. toctree::
   :maxdepth: 1

   installation.rst
   quickstart.rst
   tutorials.rst
   reference.rst

See :ref:`Citation Information <cite_info>` for how to acknowledge DSPS.
