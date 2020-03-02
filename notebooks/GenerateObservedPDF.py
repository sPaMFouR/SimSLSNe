#!/usr/bin/env python
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxx-----------Generating PDF of Observed Super Luminous Supernovae--------------xxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #

# ------------------------------------------------------------------------------------------------------------------- #
# Import Modules
# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, skewnorm
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Paths to Files and Directories
# ------------------------------------------------------------------------------------------------------------------- #
os.environ["HOME"] = "/data/asingh/simsurvey"
DIR_HOME = os.environ.get("HOME")
DIR_INPUT = os.path.join(DIR_HOME, "data/")

file_sample = os.path.join(DIR_INPUT, "ZTFSample.csv")
outfile_pdf = os.path.join(DIR_INPUT, "ZTFPDF.dat")
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
prop_df = pd.read_csv(file_sample, comment='#')
print (prop_df.shape[0])
sample = list(prop_df['gpeak_abs'].values)
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
mean = np.mean(sample)
std = np.std(sample)
skewness = skew(sample)
kurt = kurtosis(sample)
print ('Mean={0:.3f}, Std.={1:.3f}, Skew={2:.3f}, Kurtosis={3:.3f}'.format(mean, std, skewness, kurt))
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
a, loc, scale = skewnorm.fit(sample)
values = np.arange(-23.5, -19.5, 0.1)

prob_norm = [norm(mean, std).pdf(val) for val in values]
prob_skewnorm = skewnorm.pdf(values, a, loc, scale)

ascii.write([values, prob_skewnorm], outfile_pdf, names=['mag', 'prob'],
            formats={'mag': '0.2f', 'prob': '0.3f'}, overwrite=True)
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)

ax.hist(sample, bins=7, density=True, color='dimgrey')
ax.plot(values, prob_norm, color='r')
ax.plot(values, prob_skewnorm, color='navy')

plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #
