# Version 0.1
# Last Update - Feb 5, 2020

# ------------------------------------------------------------------------------------------------------------------- #
# Import Modules
# ------------------------------------------------------------------------------------------------------------------- #
import os
import glob
import pickle
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Initialize Globals
# Path to the Output Directory
# ------------------------------------------------------------------------------------------------------------------- #
gallat_cutoff = 10
mag_cutoff = 19.5
mag_cutoff_RCF = 18.5
npoints = 1

os.environ["HOME"] = "/data/asingh/simsurvey"
DIR_HOME = os.environ.get("HOME")
DIR_OUTPUT = os.path.join(DIR_HOME, "output/")

list_files = glob.glob(os.path.join(DIR_OUTPUT + '*Raw*.pkl'))
cols = ['SNType', 'Model', 'Zstart', 'Zend', 'Rate', 'Iteration', 'RawLCs', 'FilteredLCs']
logfilename = 'LogSim_ZTF.dat'
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Helper Functions
# ------------------------------------------------------------------------------------------------------------------- #

def display_text(text_to_display):
    """
    Displays text mentioned in the string 'text_to_display'
    Args:
        text_to_display : Text to be displayed
    Returns:
        None
    """
    print("# " + "-" * (12 + len(text_to_display)) + " #")
    print("# " + "-" * 5 + " " + str(text_to_display) + " " + "-" * 5 + " #")
    print("# " + "-" * (12 + len(text_to_display)) + " #\n")

# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Functions for Filtering the Simulated Light Curves Data
# ------------------------------------------------------------------------------------------------------------------- #

def check_separation(temp):
    """
    Check whether there atleast 2 points with a separation of 12 hrs in the Input Pandas DataFrame.
    Args:
        temp     : Input Pandas DataFrame for checking check_separation
    Returns:
        True     : When the simulated light curve satisfies the separation criteria
        False    : When the simulated light curve fails to satisfy the separation criterion.
    """
    time_diff = np.diff(temp['time'].values)
    points_diff = len([val for val in time_diff if abs(val) >= 0.5])
    if points_diff >= 2:
        return True
    else:
        return False


def check_indices(lc_df, filename):
    """
    Checks whether the Light Curve data in the Pandas DataFrame satisfy the Filtering Criterions.
    Args:
        lc_df    : Input Pandas DataFrame containing the monochromatic light curve
        filename : Name of the input Pickle File containing the simulated light curves
    Returns:
        True     : When the simulated light curve has to be dropped
        False    : When the simulated light curve satisfies all the filtering criterions.
    """
    # Obtain LC that have an SNR greater than 3 and 5(Essential for filtering)
    snr3_df = lc_df[lc_df['snr'] >= 3].copy()
    snr5_df = lc_df[lc_df['snr'] >= 5].copy()

    # Check whether 5 sigma LC adheres to the Apparent Magnitude cutoff of 19.5 mag
    if snr5_df[snr5_df['mag'] <= mag_cutoff].shape[0] < npoints:
        return True

    # Compute the Maximum Magnitude based on 5 sigma LC
    # Compute the Detection Epoch based on 3 sigma LC
    elif 'Magnetar' in filename:
        max_mag = snr5_df['mag'].min()
        
        # Check whether the LC obeys the RCF cut
        if max_mag <= mag_cutoff_RCF:
            return False
        else:
            # Compute the Maximum Epoch based on 5 sigma LC
            # Compute the Detection Epoch based on 3 sigma LC
            max_epoch = snr5_df.loc[snr5_df['mag'] == max_mag, 'phase'].values[0]
            det_epoch = np.min(snr3_df['phase'])
            deltat = max_epoch - det_epoch
            
            if deltat >= 21:
                return False
            elif deltat >= 8:
                det_mag = snr3_df.loc[snr3_df['phase'] == det_epoch, 'mag'].values[0]
                if det_mag >= 0.35:
                    return False   
            else: 
                return True

    # Check whether there are few detections before +50 d
    elif 'Template' in filename:
        if temp[temp['phase'] <= 50].shape[0] < npoints:
            return True
        else:
            return False
    else:
        print ("ERROR: Invalid Pickle FileName") 


def filter_lcdata(filename, inp_data):
    """
    Filter the Simulated Light Curves data from simsurvey using the criterions mentioned below.
    The light curve was filtered with the condition that SNR >= 3. The requirements are:
    1) Atleast 4 detections are brighter the ztfr-band magnitude cutoff of 19.5 mag.
    2) The 4 detections should be separated by atleast 12 hrs.
    3) There are 4 detections before +50 d.
    Args:
    filename  : Name of the input Pickle File
    inp_data  : Simulated light curves data from simsurvey
    Returns:
    inp_data  : Filtered light curves data with necessary requirements
    """
    drop_indices = []
    for lc in range(len(inp_data['lcs'])):
        lc_df = pd.DataFrame(data=inp_data['lcs'][lc], columns=inp_data['lcs'][0].dtype.names)
        lc_df['mag'] = -2.5 * np.log10(lc_df['flux']) + lc_df['zp']
        lc_df['phase'] = lc_df['time'] - inp_data['meta']['t0'][lc]
        lc_df['snr'] = lc_df['flux'] / lc_df['fluxerr']

        temp_g = lc_df.loc[lc_df['band'] == 'ztfg']
        temp_r = lc_df.loc[lc_df['band'] == 'ztfr']

        g = check_indices(temp_g, filename)
        r = check_indices(temp_r, filename)

        if g and r:
            drop_indices.append(lc)
        else:
            # Check for LCs that do not adhere to Galactic Latitude cutoff
            ra, dec = inp_data['meta']['ra'][lc], inp_data['meta']['dec'][lc]
            coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
            if abs(coord.galactic.b.degree) < gallat_cutoff:
                drop_indices.append(lc)
                        
    out_data = inp_data.copy()
    for val in sorted(drop_indices, reverse=True):
        del(out_data['lcs'][val])
    for key in [x for x in out_data.keys() if x in['meta', 'stats']]:
        for val in out_data[key].keys():
            # if not val in ['p_binned', 'mag_max']:
            if val not in ['p_binned', 'mag_max']:
                out_data[key][val] = np.delete(out_data[key][val], drop_indices, 0)
            else:
                for subkey in out_data[key][val].keys():
                    out_data[key][val][subkey] = np.delete(out_data[key][val][subkey], drop_indices, 0)

    return out_data

# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Filter the Raw Pickle Files Generated by 'simsurvey' in the OUTPUT Directory
# ------------------------------------------------------------------------------------------------------------------- #

def main():
    """
    """
    log_df = pd.DataFrame(columns=cols)
    log_df.index.name = 'Name'

    for filepath in list_files:
        filename = filepath.split('/')[-1]
        display_text("Filtering Pickle File '{0}'...".format(filename))
        data_raw = pickle.load(open(filepath, 'rb'))

        if data_raw['lcs'] is None or data_raw['lcs'] == []:
            display_text("FilteringError: Pickle File '{0}' failed to filter".format(filename))
            continue
        else:
            len_raw = len(data_raw['lcs'])
            data_mod = filter_lcdata(filename, data_raw)
            pickle.dump(data_mod, open(filepath.replace('Raw', 'Filtered'), 'wb'))
            usable_vals = filename.strip('*.pkl').split('_')[2:]
            log_df.loc[filename, cols] = usable_vals + [len_raw, len(data_mod['lcs'])]

    print (log_df)
    log_tab = Table.from_pandas(log_df)
    log_tab.write(os.path.join(DIR_OUTPUT, logfilename), format='ascii.fixed_width', delimiter=' ', overwrite=True)

# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Execute the Standalone Code
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    main()
# ------------------------------------------------------------------------------------------------------------------- #
