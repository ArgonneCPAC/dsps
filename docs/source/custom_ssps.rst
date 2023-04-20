:orphan:

.. _custom_ssp_libraries:

Acquiring and using alternative SSP libraries
==============================================
This section of the docs describes how to use DSPS with alternatives to the 
default SSP libraries supplied at the DSPS data URL.

Using python-fsps
----------------------------------

The following DSPS function implements a simple standalone 
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
    and metallicity as the mass fraction of elements heavier than helium (not in Zsun!), 
    and both quantities should be supplied in base-10 log.
    And the SSP spectra are defined in units of Lsun/Hz as a function of 
    wavelength λ in Angstroms.


