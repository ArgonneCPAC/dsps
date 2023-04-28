SSP spectra
-----------
The ssp_data_fsps_v3.2_lgmet_age.h5 file stores the default choice DSPS makes 
for the SEDs of the simple stellar populations (SSPs).
These SEDs were generated using dsps/scripts/write_fsps_data_to_disk.py, 
which calls python-fsps using its default settings
(including contributions from nebular emission).
And so the DSPS default SSP spectra are the same as those used in v3.2 of FSPS.

ssp_data_fsps_v3.2_lgmet_age.h5 is a flat hdf5 file with four columns:

* ssp_lgmet - log10(Z) grid of shape (n_met, )
* ssp_lg_age - log10(age/Gyr) grid of shape (n_age, )
* ssp_wave - λ/AA grid of shape (n_wave, )
* ssp_flux - flux in Lsun/Hz/Msun of shape (n_met, n_age, n_wave)

You can load the SSP data using the following convenience function:

>>> from dsps import load_ssp_templates
>>> ssp_data = load_ssp_templates("/path/to/dsps/data/ssp_data_fsps_v3.2_lgmet_age.h5")

As described in the Quickstart Guide on dsps.readthedocs.io,
you can set the DSPS_DRN environment variable to your default data location.

Configuring DSPS default data location
--------------------------------------
All of the functions in DSPS accept plain ndarrays as inputs,
and so whatever SED library you choose,
you can store these ndarrays wherever you like and in whatever format you prefer.
As a convenience, you can set up DSPS to remember where you store these data 
by setting an environment variable DSPS_DRN to their location on disk.

To do this in bash:

    export DSPS_DRN="/path/to/dsps/data"

With the DSPS_DRN environment variable defined as above,
you can now call the `dsps.load_ssp_templates` function without any arguments.

Filter transmission curves
--------------------------
Transmission curve data are stored as a flat numpy structured array with two columns:

* wave - λ/AA grid of shape (n_trans, )
* transmission - transmission curve of the filter, shape (n_trans, )

You can load filter transmission curves using the following convenience function:

>>> from dsps.data_loaders import load_transmission_curve
>>> trans_curve = load_transmission_curve("/path/to/dsps/data/lsst_r*")

If you have set the DSPS_DRN environment variable to your default data location,
then you will need to store transmission curves in the DSPS_DRN/filters subdirectory.
See the Quickstart Guide on dsps.readthedocs.io for more information.
