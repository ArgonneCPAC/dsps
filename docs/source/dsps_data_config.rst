:orphan:

.. _dsps_drn_config:

Configuring your default DSPS data location
-------------------------------------------

Whichever SED libraries you choose, you can optionally set an environment variable
**DSPS_DRN** with the default location of the data you use with DSPS.
To do that in bash:

.. code-block:: bash

    export DSPS_DRN="/path/to/dsps/data"

If the **DSPS_DRN** environment variable has been set to the disk location storing
the SSP data, then you can load the data like this:

.. code-block:: python

    >>> from dsps import load_ssp_templates
    >>> ssp_data = load_ssp_templates()

Without the **DSPS_DRN** environment variable, you will just need to pass the path of the 
data to the **load_ssp_templates** function. The data returned by this function is simply a 
sequence of ndarrays, and so you can also elect to ignore the data-loading
convenience function if you prefer to keep track of these individual arrays yourself.


Loading filter transmission curves
----------------------------------

The filters directory at 
`this URL <https://portal.nersc.gov/project/hacc/aphearin/DSPS\_data/>`__ 
stores a few filter transmission curves to help you quickly get started 
with photometry predictions. For a more comprehensive set of filters,
see the `kcorrect library <https://github.com/blanton144/kcorrect/tree/main/python/kcorrect/data/responses>`__.

DSPS comes with a convenience function for loading filter transmission curves:

.. code-block:: python

    >>> from dsps.data_loaders import load_transmission_curve
    >>> trans_curve = load_transmission_curve("/path/to/dsps/data/lsst_r*")

In order to use this function together with the **DSPS_DRN** environment variable, 
you will need to store your transmission curve data in 
a subdirectory of your default data location: **DSPS_DRN/filters**.
Just as with the SSP data, the data-loading convenience function 
just returns a sequence of ndarrays, and so you are free to ignore this function 
if you prefer to manage the data storing your transmission curves.