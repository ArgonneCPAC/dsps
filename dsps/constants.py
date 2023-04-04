"""
"""
import numpy as np

# z=0 age of the universe for default cosmology
TODAY = 13.79
LGT0 = np.log10(TODAY)

# Constants related to SFH integrals
SFR_MIN = 1e-14
T_BIRTH_MIN = 0.001
N_T_LGSM_INTEGRATION = 100

# Constants related to metallicity weights
LGMET_LO, LGMET_HI = -10.0, 10.0
