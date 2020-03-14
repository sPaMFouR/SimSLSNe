# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxx-----------Generating Distribution of Magnetar Model Parameters--------------xxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #

# ------------------------------------------------------------------------------------------------------------------- #
# Import Modules
# ------------------------------------------------------------------------------------------------------------------- #
import os
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as st
from astropy.io import ascii
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Paths To Files and Directories
# ------------------------------------------------------------------------------------------------------------------- #
os.environ["HOME"] = "/data/asingh/simsurvey"
DIR_HOME = os.environ.get("HOME")
DIR_INPUT = os.path.join(DIR_HOME, "data/")

bins = 6
file_sample = os.path.join(DIR_INPUT, "ZTFSample.csv")
outfile_pdf = os.path.join(DIR_INPUT, "ZTFPDF_{0}.dat".format(bins))

prop_df = pd.read_csv(file_sample, comment='#')
print (prop_df.shape[0])
sample = list(prop_df['gpeak_abs'].values)
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Fit The Best Step Wise PDF
# ------------------------------------------------------------------------------------------------------------------- #
hist = np.histogram(sample, bins=bins, density=True)
hist_dist = st.rv_histogram(hist)
magarr = np.arange(-23.5, -19.5, 0.05)

ascii.write([magarr, hist_dist.pdf(magarr)], outfile_pdf, names=['mag', 'prob'],
            formats={'mag': '0.2f', 'prob': '0.3f'}, overwrite=True)
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Plot the Step Wise PDF 
# ------------------------------------------------------------------------------------------------------------------- #
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

ax.hist(sample, bins=bins, density=True, alpha=0.5, label='Data')
ax.plot(magarr, hist_dist.pdf(magarr), color='k', label='PDF')

ax.legend(fontsize=15)
ax.set_title('Probability Distribution Function [Bins = {0}]'.format(bins))
ax.set_xlabel('Peak ztfg-Band Magnitudes [mag]')
ax.set_ylabel('Frequency')

fig.savefig('PLOT_StepWisePDF_{0}.pdf'.format(bins), format='pdf', dpi=2000, bbox_inches='tight')
# plt.show()
plt.close(fig)
# ------------------------------------------------------------------------------------------------------------------- #
