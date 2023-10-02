# SimSLSNe

[![Build Status](https://img.shields.io/badge/version-0.1-orange)](https://github.com/sPaMFouR/RedPipe)
[![Python 3.7](https://img.shields.io/badge/python-3.7.2-brightgreen.svg)](https://www.python.org/downloads/release/python-372/)

Simulation code for estimating Volumetric Rates of Super-Luminous Supernovae (SLSNe) using data from Zwicky Transient Facility (ZTF). The code is built over the Simulation tool [simsurvey](https://github.com/ZwickyTransientFacility/simsurvey) developed by Ulrich Feindt (see Documentation [here](https://simsurvey.readthedocs.io/) and Paper [here](https://arxiv.org/abs/1902.03923)).

###Unique Pointings of ZTF (2018-2020)
![Unique Pointings of ZTF (2018-2020)](plots/PLOT_ZTFUniquePointings_3DayBarV.png)
###Simulated Set of LCs from ZTF
![Simulated Set of LCs from ZTF)](plots/PLOT_LCS_Magnetar_0.07_0.4_10.0_5.png)
###Sample Simulated Light Curve and Parameter Extraction
![Sample Simulated Light Curve and Parameter Extraction](plots/PLOT_LC-38_Magnetar_0.06_0.2_5.0_0.png)

The code is still under development.

Authors
-------

* **Avinash Singh** (IIA, Bengaluru)
* **Lin Yan** (Caltech, Pasadena)


Requirements
-------

- astropy
- sncosmo
- simsurvey
- matplotlib, seaborn (for plotting)
