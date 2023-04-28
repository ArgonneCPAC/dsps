Quickstart Guide
================
This guide is meant to help you quickly get up and running with DSPS.
You can find more in-depth information in the 
:ref:`Tutorials <tutorials>` section of the documentation.


Downloading the default SED Library
-----------------------------------
The basis of DSPS predictions for the SED of a galaxy
is an underlying spectral library of simple stellar populations (SSPs).
You can download the default SEDs used by DSPS at
`this URL <https://portal.nersc.gov/project/hacc/aphearin/DSPS\_data/>`__.

Once you have downloaded these data, you can load them using
the following convenience function:

.. code-block:: python

    >>> from dsps import load_ssp_templates
    >>> ssp_data = load_ssp_templates(fn="/path/to/dsps/data/fname.h5")

See :ref:`dsps_drn_config` for instructions on how to set a default location
for your DSPS data, including both your SSP library
as well as filter transmission curves.

There are many other sources of SED libraries that are publicly available,
and you can use DSPS with whatever SED library is most appropriate for
your science application. 
See :ref:`Using Alternative SED libraries <custom_ssp_libraries>` for further information.

Demo Notebook
-----------------------------------
Once you're set up with an SSP library, you can follow the notebook below
for a code demo showing how to use the core functions in DSPS.

.. toctree::
   :maxdepth: 1
   
   dsps_quickstart.ipynb