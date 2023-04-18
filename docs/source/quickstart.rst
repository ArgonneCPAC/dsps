Quickstart Guide
================

Getting Started
---------------
You can use DSPS to calculate many common quantities of stellar population synthesis,
but the DSPS package does not include data storing SED libraries,
and so you'll need to download these.
To quickly get up and running, you can download the default SEDs used by DSPS at
`this URL <https://portal.nersc.gov/project/hacc/aphearin/DSPS\_data/>`__.
DSPS includes a convenience function for using 
`python-fsps <https://dfm.io/python-fsps/current/>`__
to generate customizable versions of these SED libraries.
There are many sources of SED libraries that are publicly available.
You can use DSPS with whatever SED library is most appropriate for
your science application.

.. toctree::
   :maxdepth: 1
   
   dsps_quickstart.ipynb