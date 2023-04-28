:orphan:

.. _custom_ssp_libraries:

Acquiring and using alternative SSP libraries
==============================================
This section of the docs describes how to use DSPS with alternatives to the 
default SSP libraries supplied at the DSPS data URL.


.. _dsps_drn_config:

Configuring your default SSP data location
------------------------------------------

Whichever SED libraries you choose, you can optionally set an environment variable
DSPS_DRN with the default location of the data you use with DSPS.
To do that in bash:

.. code-block:: bash

    export DSPS_DRN="/path/to/dsps/data"

If the DSPS_DRN environment variable has been set to the disk location storing
the SSP data, then you can load the data like this:

.. code-block:: python

    >>> from dsps import load_ssp_templates
    >>> ssp_data = load_ssp_templates()

Without the DSPS_DRN environment variable, you will just need to pass the path of the 
data to the load_ssp_templates function. The data returned by this function is simply a 
sequence of ndarrays, and so you can also elect to ignore the data loader 
convenience function if you prefer to keep track of these individual arrays yourself.


Using python-fsps
----------------------------------

DSPS includes a convenience function for using 
`python-fsps <https://dfm.io/python-fsps/current/>`__
to generate customizable versions of these SED libraries.
The following function implements a simple standalone 
wrapper around python-fsps that returns SSP information 
in a convenient form of arrays and matrices for use with the rest of DSPS.

.. code-block:: Python

    from dsps.data_loaders import retrieve_ssp_data_from_fsps

With python-fsps there are many other options available for 
generating a library of SSP spectra. 
The code in this function is straightforward 
to adapt to generate alternative SSP templates.

Using spectra from another library
----------------------------------

.. Important:: Mind your units and logarithms when using your own SSP library.
    As we will see in the demo notebook and the docstrings of the source code,
    stellar ages should be specified in units of Gyr (not in yr!)
    and metallicity as the mass fraction of elements heavier than helium 
    (not noramlized by Zsun!), and both quantities should be supplied in base-10 log.
    The SSP spectra are defined in units of Lsun/Hz as a function of 
    wavelength λ in Angstroms.

There are many other sources of SED libraries that are publicly available,
and you can use DSPS with whatever SED library is most appropriate for
your science application. As described above, there is also no need to use the 
data-loading functions in DSPS, since these are just convenience functions 
that return a sequence of flat ndarrays.