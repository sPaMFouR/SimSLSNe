# Version 0.1
# Last Update - Feb 7, 2020

# ------------------------------------------------------------------------------------------------------------------- #
# Import Modules
# ------------------------------------------------------------------------------------------------------------------- #
import os
import sys
import time
import pickle
import logging
import datetime
import warnings
import numpy as np
import pandas as pd

import sncosmo
import simsurvey
from astropy.time import Time
from astropy import units as u
from argparse import ArgumentParser
from scipy.integrate import cumtrapz
from astropy.cosmology import Planck15
from astropy.coordinates import SkyCoord
from scipy.interpolate import InterpolatedUnivariateSpline as Spline1d

warnings.filterwarnings('ignore')
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Initialize Globals
# ------------------------------------------------------------------------------------------------------------------- #
dict_bands = {'ztfg': [4087, 4722.7, 5522], 'ztfr': [5600, 6339.6, 7317], 'desi': [7317, 7886.1, 8884]}
dict_rlambda = {'ztfg': 3.694, 'ztfr': 2.425, 'desi': 1.718}
params_magnetar = ['P', 'B', 'Mej', 'Vej', 'kappa', 'kappa_gamma', 'Mns', 'Tf']
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Initialize Directories
# ------------------------------------------------------------------------------------------------------------------- #
os.environ["HOME"] = "/data/asingh/simsurvey"
DIR_HOME = os.environ.get("HOME")

# Define the Input Directory
DIR_INPUT = os.path.join(DIR_HOME, "data/")

# Define the Output Directory
DIR_OUTPUT = os.path.join(DIR_HOME, "output/")

# Directory containing the dust map files of Schlegel, Finkbeiner & Davis (1998)
DIR_SFD = os.path.join(DIR_HOME, "data/sfd98")

# ZTF Pointing File
DIR_DATA = "/data/cfremling/simsurvey"
survey_file = os.path.join(DIR_DATA, "notebooks/df_sim_stats_full.p")

# Name of the File containing the Template Light Curve
file_template = os.path.join(DIR_INPUT, "PTF12dam.dat")
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Helper Functions
# ------------------------------------------------------------------------------------------------------------------- #

def abmag_to_flambda(abmag, band):
    return 2.99792e18 * (10 ** (-0.4 * (abmag + 48.6))) / (dict_bands[band][1] ** 2)


def z_to_distmod(z):
    distance = Planck15.luminosity_distance(z).value
    return 5 * np.log10(distance * 1e6) - 5


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
# Observed Template Function
# ------------------------------------------------------------------------------------------------------------------- #

def ObservedTemplateSource(input_lc):
    """
    Make a TimeSeriesSource of the observed photometric template assuming a certain SED shape.
    Args:
        input_lc   : Input light curve template to be used as input to the simulation
    Returns:
        model      : A sncosmo.Source that specifies a spectral timeseries of the template
    """
    def get_waverange(band):
        limits = dict_bands[band]
        if band == 'ztfg':
            waverange = np.arange(limits[0] - 1500, limits[-1], 10)
        elif band == 'ztfr':
            waverange = np.arange(limits[0], limits[-1], 10)
        elif band == 'desi':
            waverange = np.arange(limits[0], limits[-1] + 2000, 10)
        return waverange

    lc = pd.read_csv(input_lc, sep='\s+', comment='#')
    lc['flux_g'] = lc['M_g'].apply(lambda x: abmag_to_flambda(x, 'ztfg'))
    lc['flux_r'] = lc['M_r'].apply(lambda x: abmag_to_flambda(x, 'ztfr'))
    lc['flux_i'] = lc['M_i'].apply(lambda x: abmag_to_flambda(x, 'desi'))
    lc['Phase'] = lc['Phase'] * 2

    norm_factor = lc['flux_r'].max()
    lc[['flux_g', 'flux_r', 'flux_i']] /= norm_factor

    phase = lc['Phase'].values
    masterwave = np.array([val for band in dict_bands.keys() for val in get_waverange(band)])

    masterflux = []
    for epoch in phase:
        temp = lc.loc[lc['Phase'] == epoch]
        tempflux = []
        for band, limits in dict_bands.items():
            wave = get_waverange(band)
            tempflux.extend([temp['flux_' + band[-1]].values[0]] * len(wave))
        masterflux.append(tempflux)
    masterflux = np.array(masterflux)

    source = sncosmo.TimeSeriesSource(name='SLSN', phase=phase, wave=masterwave, flux=masterflux, zero_before=True)

    return source

# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Class for loading Raw ZTF Data
# ------------------------------------------------------------------------------------------------------------------- #

class Load_ZTFData():
    """
    Load ZTF Data Files for Simulation.
    """

    def __init__(self):
        dir_ZTF = os.path.dirname(DIR_INPUT)
        if not os.path.exists(dir_ZTF):
            display_text("ERROR: Directory With ZTF Camera and Pointing Information Missing")
            sys.exit(1)

        dir_out = os.path.dirname(DIR_OUTPUT)
        if not os.path.exists(dir_out):
            display_text("Output Directory Doesn't Exist. Creating one...")
            os.makedirs(dir_out)

        dir_sfd = os.path.dirname(DIR_SFD)
        if not os.path.exists(dir_sfd):
            display_text("ERROR: Directory Containing Dust Map Files Not Found")
            sys.exit(1)

    def load_fields_ccd(self, DIR_INPUT):
        """
        Load ZTF Fields and Corners Data.
        """
        fields_file = DIR_INPUT + 'ZTF_Fields.txt'
        if not os.path.isfile(fields_file):
            display_text("ERROR: File 'ZTF_Fields.txt' Missing")
            sys.exit(1)

        ccd_file = DIR_INPUT + 'ZTF_corners.txt'
        if not os.path.isfile(ccd_file):
            display_text("ERROR: File 'ZTF_corners.txt' Missing")
            sys.exit(1)

        raw_fields = np.genfromtxt(fields_file, comments='%')
        fields = {'field_id': np.array(raw_fields[:, 0], dtype=int),
                  'ra': raw_fields[:, 1], 'dec': raw_fields[:, 2]}
        fields_df = pd.DataFrame(fields)

        ccd_corners = np.genfromtxt(ccd_file, skip_header=1)
        ccds = [ccd_corners[4 * k:4 * k + 4, :2] for k in range(16)]

        return fields, ccds

    def load_ztf_bands(self, dir_bandpass=DIR_INPUT):
        """
        Load ZTF filters on to the sncosmo registry.
        """
        bands = {'ztfg': 'ztfg_eff.txt', 'ztfr': 'ztfr_eff.txt', 'ztfi': 'ztfi_eff.txt'}

        for band in bands.keys():
            filename = os.path.join(DIR_INPUT, bands[band])
            bpass = np.loadtxt(os.path.join(dir_bandpass, filename))
            bandname = sncosmo.Bandpass(bpass[:, 0], bpass[:, 1], name=band)
            sncosmo.registry.register(bandname, force=True)

    def load_modified_input(self, filename, clean=True):
        """
        Load and Modify the input data as Pandas DataFrame to be passed to the SimSurvey.
        Args:
        filename : Input file with the raw ZTF data
        Returns:
        out_df   : Output DataFrame with relevant modifications
        """
        dict_filterid = {1: 'ztfg', 2: 'ztfr', 3: 'desi'}
        survey_start = Time("2018-03-17 00:00:00.000").jd
        survey_end = Time("2019-12-31 00:00:00.000").jd

        inp_df = pd.read_pickle(filename)
        out_df = inp_df.copy()
        out_df['filterid'] = out_df['filterid'].apply(lambda band: dict_filterid[band])
        out_df['skynoise'] = [(10 ** (-0.4 * (limmag - 30))) / 5. for limmag in out_df['limMag']]

        if not clean:
            return out_df
        else:
            out_df = out_df[~out_df['fieldid'].isin([880, 881])]
            out_df = out_df[out_df['jd'] > survey_start][out_df['jd'] < survey_end]
            display_text("Survey pointings for All ZTF Programs: {0}".format(len(out_df)))
            display_text("Survey pointings for MSIP Programs: {0}".format(len(out_df[out_df['progid'] == 1])))
            return out_df

# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Class for running the Simulation
# ------------------------------------------------------------------------------------------------------------------- #

class RunSim(Load_ZTFData):
    """
    """
    def __init__(self):
        Load_ZTFData.__init__(self)
        self.days_before_start = 30
        self.days_after_end = 30

        logging.info('Days before Survey Start: {0}'.format(self.days_before_start))
        logging.info('Days after Survey End: {0}'.format(self.days_after_end))

    def make_file_name(self, args, rate, iteration):
        if not os.path.exists(DIR_OUTPUT):
            os.makedirs(DIR_OUTPUT)

        file_name = "LCS_Raw_SLSN_Template_{1}_{2}_{3}_{4}.pkl".format(args.redshift[0],
                                                                  args.redshift[1], rate, iteration)
        return DIR_OUTPUT + file_name

    def run_observedtemp(self, raw_df, fields, ccds, rate, args, iteration):
        logging.info('zmin: %f ', args.redshift[0])
        logging.info('zmax:  %f ', args.redshift[1])
        logging.info('Template: %s ', 'Template')
        logging.info('Rate: %f ', rate * 1e-7)

        # Create the Model and combine it with propagation effects
        source = ObservedTemplateSource(file_template)
        dust = sncosmo.CCM89Dust()
        model = sncosmo.Model(source=source, effects=[dust], effect_names=['host'], effect_frames=['rest'])

        # Randomly draw the peak magnitude of SLSN
        def random_parameters(redshifts, model, mag=(-21.642, 0), r_v=2., ebv_rate=0.11, cosmo=Planck15, **kwargs):
            """
            """
            amp = []
            out = {}
            for z in redshifts:
                mabs = np.random.normal(mag[0], mag[1])
                model.set(z=z)
                model.set_source_peakabsmag(absmag=mabs, band='ztfr', magsys='ab', cosmo=cosmo)
                amp.append(model.get('amplitude'))

            out['amplitude'] = np.array(amp)

            return out

        # Load Input Data into the Survey Plan
        plan = simsurvey.SurveyPlan(time=raw_df['jd'], band=raw_df['filterid'], obs_field=raw_df['fieldid'],
                                    skynoise=raw_df['skynoise'], obs_ccd=raw_df['chid'], comment=raw_df['progid'],
                                    ccds=ccds, fields={k: v for k, v in fields.items()
                                                       if k in ['ra', 'dec', 'field_id', 'width', 'height']})

        mjd_range = (plan.cadence['time'].min() - self.days_before_start,
                     plan.cadence['time'].max() + self.days_after_end)

        # Input to the Transient Generator
        tr = simsurvey.get_transient_generator((args.redshift[0], args.redshift[1]),
                                               ratefunc=lambda z: rate * 1e-7,
                                               dec_range=(-31, 90),
                                               mjd_range=(mjd_range[0], mjd_range[1]),
                                               sfd98_dir=DIR_SFD,
                                               transientprop={'lcmodel': model, 'lcsimul_func': random_parameters})

        # Simulate Survey
        survey = simsurvey.SimulSurvey(generator=tr, plan=plan)
        lcs = survey.get_lightcurves(progress_bar=True, notebook=False)

        output_filename = self.make_file_name(args, rate, iteration)
        lcs.save(output_filename)
        logging.info('Filename %s', output_filename)


def main():
    """
    Run the Simulation with inputs from command line arguments
    """
    logging.basicConfig(filename="SimZTF.log", level=logging.DEBUG)
    logging.info('Date: %s ', datetime.datetime.now())

    parser = ArgumentParser(description='ZTF Rate simulation for SLSN')

    parser.add_argument('-z', '--redshift', nargs=2, type=float, default=[0.07, 0.3],
                        help="Redshift Range, [Default = [0.07, 0.3]")

    parser.add_argument('-ratetype', type=str, default='Single',
                        help="Simulation Input ('Single', 'Multiple'), [Default = 'Single']")

    parser.add_argument('-rate', type=float, default=3.5,
                        help="Rate Range for SN in terms of 10^-7 / yr / Mpc [Default = 3.5]")

    parser.add_argument('-raterange', nargs=3, type=float, default=[0.5, 3.6, 0.75],
                        help="Rate Range for SN in terms of 10^-7 / yr / Mpc, [Default = [0.5, 3.6, 0.75]]")

    parser.add_argument('-runs', type=int, default=1,
                        help="Run Simulation 'runs' times for each Rate value")

    parser.add_argument('-data_file', type=str, default=survey_file,
                        help="ZTF Input Filename")

    args = parser.parse_args()

    run_Sim = RunSim()

    def runiter(rate):
        for iteration in range(args.runs):
            display_text("Iteration {0}: z = [{1:.2f}, {2:.2f}], Rate = {3:.2f}e-7".format(iteration, args.redshift[0],
                                                              				 args.redshift[1], rate))
            start_time = time.time()
            run_Sim.run_observedtemp(df, fields, ccds, rate, args, iteration)
            display_text("Iteration {0}: Time = {1:.2f} seconds".format(iteration, time.time() - start_time))

    # Load ZTF data
    data_loader = Load_ZTFData()
    data_loader.load_ztf_bands()
    fields, ccds = data_loader.load_fields_ccd(DIR_INPUT)

    df = data_loader.load_modified_input(survey_file)
    logging.info('Input Data File: %s ', survey_file)

    # Run Simulation
    if args.ratetype == 'Single':
        runiter(args.rate)
    elif args.ratetype == 'Multiple': 
        for rate in np.arange(args.raterange[0], args.raterange[1], args.raterange[2]):
            runiter(rate)
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Execute the Standalone Code
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    main()
# ------------------------------------------------------------------------------------------------------------------- #
