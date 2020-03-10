#!/usr/bin/env python
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxx-----------Generating PDF of Observed Super Luminous Supernovae--------------xxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #

# ------------------------------------------------------------------------------------------------------------------- #
# Import Modules
# ------------------------------------------------------------------------------------------------------------------- #
import os
import matplotlib
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, skewnorm

matplotlib.style.use('ggplot')
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Paths to Files and Directories
# ------------------------------------------------------------------------------------------------------------------- #
os.environ["HOME"] = "/data/asingh/simsurvey"
DIR_HOME = os.environ.get("HOME")
DIR_INPUT = os.path.join(DIR_HOME, "data/")

file_sample = os.path.join(DIR_INPUT, "ZTFSample.csv")
outfile_pdf = os.path.join(DIR_INPUT, "ZTFPDF.dat")

prop_df = pd.read_csv(file_sample, comment='#')
print (prop_df.shape[0])
sample = list(prop_df['gpeak_abs'].values)
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Calculate the Mean, Std, Skew and Kurtosis of the Distribution
# ------------------------------------------------------------------------------------------------------------------- #
mean = np.mean(sample)
std = np.std(sample)
skewness = skew(sample)
kurt = kurtosis(sample)
print ('Mean={0:.3f}, Std.={1:.3f}, Skew={2:.3f}, Kurtosis={3:.3f}'.format(mean, std, skewness, kurt))
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Fit the Skewed Probability Distribution to the ZTF Sample
# ------------------------------------------------------------------------------------------------------------------- #
a, loc, scale = skewnorm.fit(sample)
magarr = np.arange(-23.5, -19.5, 0.05)

prob_norm = [norm(mean, std).pdf(mag) for mag in magarr]
prob_skewnorm = skewnorm.pdf(magarr, a, loc, scale)

ascii.write([magarr, prob_skewnorm], outfile_pdf, names=['mag', 'prob'],
            formats={'mag': '0.2f', 'prob': '0.3f'}, overwrite=True)
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Plot the Best Fit Skewed Gaussian PDF
# ------------------------------------------------------------------------------------------------------------------- #
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

ax.hist(sample, bins=int(np.sqrt(prop_df.shape[0])), density=True, color='dimgrey')
ax.plot(magarr, prob_norm, color='r')
ax.plot(magarr, prob_skewnorm, color='navy')

ax.legend(fontsize=15)
ax.set_title('Probability Distribution Function')
ax.set_xlabel('Peak ztfg-Band Magnitudes [mag]')
ax.set_ylabel('Frequency')

fig.savefig('PLOT_SkewedGaussianPDF.pdf', format='pdf', dpi=2000, bbox_inches='tight')
# plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #
