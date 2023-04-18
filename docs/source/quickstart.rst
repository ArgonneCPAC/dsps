Quickstart Guide
================
This guide is meant to help you quickly get up and running with DSPS.
You can find more in-depth information in the 
:ref:`Tutorials <tutorials>` section of the documentation.


Downloading the default SED Library
-----------------------------------
The basis of DSPS predictions for the SED of a galaxy
is an underlying spectral library of simple stellar populations (SSPs).
The code in the quickstart guide is written using dummy spectra,
but in a scientific application you'll need to select a real spectral library. 
To quickly get up and running, you can download the default SEDs used by DSPS at
`this URL <https://portal.nersc.gov/project/hacc/aphearin/DSPS\_data/>`__.

DSPS includes a convenience function for using 
`python-fsps <https://dfm.io/python-fsps/current/>`__
to generate customizable versions of these SED libraries.
There are many other sources of SED libraries that are publicly available,
and you can use DSPS with whatever SED library is most appropriate for
your science application.

Whichever SED libraries you choose, you can optionally set an environment variable
DSPS_DRN with the default location of the data you use with DSPS.

.. toctree::
   :maxdepth: 1
   
   dsps_quickstart.ipynb