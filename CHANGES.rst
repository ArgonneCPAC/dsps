0.4.1 (2025-02-03)
-------------------
- Fix bug in mass-metallicity scaling relation (https://github.com/ArgonneCPAC/dsps/pull/98)


0.4.0 (2025-01-21)
-------------------
- Require python>=3.11 (https://github.com/ArgonneCPAC/dsps/pull/95)


0.3.7 (2024-02-13)
-------------------
- Require python>=3.9
- Change API of diffburst (https://github.com/ArgonneCPAC/dsps/pull/82)


0.3.6 (2024-01-09)
-------------------
- Adopt custom trapezoidal integration to calculate stellar age weights (https://github.com/ArgonneCPAC/dsps/pull/78)


0.3.5 (2023-10-14)
-------------------
- mzr model now return absolute metallicity rather than Z/Zsun (https://github.com/ArgonneCPAC/dsps/pull/75)


0.3.4 (2023-10-03)
-------------------
- Remove dsps.metallicity.mzrDEFAULT_MZR_PDICT and replace with DEFAULT_MET_PDICT (https://github.com/ArgonneCPAC/dsps/pull/71)


0.3.3 (2023-08-29)
-------------------
- Improve numerical robustness of gradients of exponentials (https://github.com/ArgonneCPAC/dsps/pull/69)


0.3.2 (2023-08-01)
-------------------
- Fix bug leading to edge case in which FSPS is assumed to be installed and configured (https://github.com/ArgonneCPAC/dsps/pull/68)


0.3.1 (2023-05-02)
-------------------
- Include .git_archival.txt and .gitattributes to help with versioning (https://github.com/ArgonneCPAC/dsps/pull/61)


0.3.0 (2023-05-01)
-------------------
- Package DSPS using pyproject.toml (https://github.com/ArgonneCPAC/dsps/pull/43)
- Remove Ode0 from flat wCDM cosmology params  (https://github.com/ArgonneCPAC/dsps/pull/39)
- Add Lower+22 unubscured fraction feature to dust attenuation (https://github.com/ArgonneCPAC/dsps/pull/37)
- Add readthedocs (https://github.com/ArgonneCPAC/dsps/pull/25)
- Add diffburst feature into dsps.experimental (https://github.com/ArgonneCPAC/dsps/pull/22)
- Reorganize top-level modules into subdirectories (https://github.com/ArgonneCPAC/dsps/pull/21)
- Remove prototype Diffstar model and all associated functions (https://github.com/ArgonneCPAC/dsps/pull/18)


0.2.0 (2022-09-09)
------------------
- First release