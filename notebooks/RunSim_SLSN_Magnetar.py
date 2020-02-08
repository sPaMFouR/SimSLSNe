# Version 0.1

# ------------------------------------------------------------------------------------------------------------------- #
# Import Modules
# ------------------------------------------------------------------------------------------------------------------- #
import os
import time
import pickle
import logging
import datetime
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
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Initialize Globals
# ------------------------------------------------------------------------------------------------------------------- #
dict_bands = {'ztfg': [4087, 4722.7, 5522], 'ztfr': [5600, 6339.6, 7317], 'desi': [7317, 7886.1, 8884]}
dict_rlambda = {'ztfg': 3.694, 'ztfr': 2.425, 'desi': 1.718}
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

# File containing the parameters for the Magnetar Model
file_param = os.path.join(DIR_INPUT, "Nicholl_Magnetar.dat")
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Class for Magnetar Source [Based on Nicholl et al. (2017)]
# ------------------------------------------------------------------------------------------------------------------- #

def blackbody(wl, T=6e3, R=1., d=1e-5):
    # wl in Ångströms
    # T in Kelvin
    # R in cm
    # d in Mpc
    # output in erg s^-1 cm^-2 Angstrom^-1
    B = 1.19104295262e+27 * (wl ** -5) / (np.exp(1.43877735383e+8 / (wl * T)) - 1)
    return np.pi * B * (R / (d * 3.0857e24)) ** 2


class MagnetarSource(sncosmo.Source):
    """Units:
    distance    -- Mpc
    P           -- ms
    B           -- Gauss
    Mej        -- M_sun
    Vej        -- 10^3 km/s
    Mns        -- M_sun
    kappa       -- cm^2/g
    kappa_gamma -- cm^2/g
    Tf         -- K
    """
    _param_names = ['distance', 'P', 'B', 'Mej', 'Vej', 'Mns', 'kappa', 'kappa_gamma', 'Tf']
    param_names_latex = ['d', 'P', 'B', 'M_{ej}', 'v_{ej}', 'M_{NS}', '\kappa', '\kappa_\gamma', 'T_f']

    def __init__(self, name=None, version=None, minphase=0.01, maxphase=300., phase_sampling=1000):
        self.name = name
        self.version = version
        self._minphase = minphase
        self._maxphase = maxphase
        self._phase_sampling = phase_sampling
        self._parameters = np.array([1e-5, 0.7, 0.01, 0.1, 14.7, 1.4, 0.05, 0.01, 6e3])
        self._set_L_out()

    def minwave(self):
        return 1e-100

    def maxwave(self):
        return 1e100

    def minphase(self):
        return self._minphase

    def maxphase(self):
        return self._maxphase

    def _F_mag(self, t):
        """Eqn (2) of Nicholl et al. 2017 - Functional form of Magnetic Dipole Radiation"""
        return (self._E_mag / self._t_mag) * (1 / (1 + t / self._t_mag) ** 2)

    @property
    def _E_mag(self):
        """Eqn (3) of Nicholl et al. 2017"""
        return 2.6e52 * ((self._parameters[5] / 1.4) ** 1.5) * (self._parameters[1] ** (-2))

    @property
    def _t_mag(self):
        """Eqn (4) of Nicholl et al. 2017"""
        return 1.3e5 * ((self._parameters[5] / 1.4) ** 1.5) *\
            (self._parameters[1] ** 2) * ((self._parameters[2] / 1e14) ** (-2))

    @property
    def _t_diff(self):
        """Eqn (6) of Nicholl et al. 2017"""
        return 9.84e6 * (self._parameters[6] * self._parameters[3] / self._parameters[4]) ** 0.5

    @property
    def _A(self):
        """Eqn (7) Of Nicholl et al. 2017"""
        return 4.77e16 * self._parameters[7] * self._parameters[3] / (self._parameters[4] ** 2)

    @property
    def _is_L_out_set(self):
        return np.all(self._parameters[1:] == self._L_out_parameters)

    def _set_L_out(self):
        """
        Eqn (5) of Nicholl et al. 2017 - Solving the Integral using scipy.integrate.cumtrapz
        (cumulative trapezoidal) and then interpolating between the steps.
        """
        t = np.linspace(self._minphase, self._maxphase, self._phase_sampling) * 86400
        y = 2 * self._F_mag(t) * np.exp((t / self._t_diff) ** 2) * (t / (self._t_diff ** 2))

        y_int = np.append([0], cumtrapz(y, t))
        L_out_steps = y_int * np.exp(-(t / self._t_diff) ** 2) * (1 - np.exp(-self._A * (t ** -2)))

        self._L_out = Spline1d(t / 86400, L_out_steps)
        self._L_out_parameters = self.parameters[1:]

    def _T_from_L(self, phase):
        """Term in Eqns (8) and (9) of Nicholl et al. 2017 that is compared to Tf"""
        return (self._L_out(phase) / (7.1256e-4 * (self._parameters[4] * phase * 8.64e12) ** 2)) ** 0.25

    def temperature(self, phase):
        """Eqn (8) of Nicholl et al. 2017"""
        if not self._is_L_out_set:
            self._set_L_out()

        T = self._T_from_L(phase)

        try:
            T[T <= self._parameters[8]] = self._parameters[8]
        except TypeError:
            if T <= self._parameters[8]:
                return self._parameters[8]
        return T

    def radius(self, phase):
        """Eqn (9) of Nicholl et al. 2017 in cm"""
        if not self._is_L_out_set:
            self._set_L_out()

        R = self._parameters[4] * phase * 8.64e12
        T = self._T_from_L(phase)
        mask = T <= self._parameters[8]

        try:
            R[mask] = (self._L_out(phase[mask]) / (7.1256e-4 * self._parameters[8] ** 4)) ** 0.5
        except TypeError:
            if T <= self._parameters[8]:
                return (self._L_out(phase) / (7.1256e-4 * self._parameters[8] ** 4)) ** 0.5
        return R

    def luminosity(self, phase):
        """Luminosity from Stefan-Boltzman law using Eqns (8) and (9) in erg/s"""
        if not self._is_L_out_set:
            self._set_L_out()

        return 7.1256e-4 * (self.temperature(phase) ** 4) * (self.radius(phase) ** 2)

    def _flux(self, phase, wave):
        """"""
        if not self._is_L_out_set:
            self._set_L_out()

        wave = np.array(wave)
        return np.array([blackbody(wave, T=self.temperature(p_), R=self.radius(p_), d=self._parameters[0])
                         for p_ in phase])

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
            print("ERROR: Directory With ZTF Camera and Pointing Information Missing")
            sys.exit(1)

        dir_out = os.path.dirname(DIR_OUTPUT)
        if not os.path.exists(dir_out):
            print("Output Directory Doesn't Exist. Creating one...")
            os.makedirs(dir_out)

        dir_sfd = os.path.dirname(DIR_SFD)
        if not os.path.exists(dir_sfd):
            print("ERROR: Directory Containing Dust Map Files Not Found")
            sys.exit(1)

    def load_fields_ccd(self, DIR_INPUT):
        """
        Load ZTF Fields and Corners Data.
        """
        fields_file = DIR_INPUT + 'ZTF_Fields.txt'
        if not os.path.isfile(fields_file):
            print("ERROR: File 'ZTF_Fields.txt' Missing")
            sys.exit(1)

        ccd_file = DIR_INPUT + 'ZTF_corners.txt'
        if not os.path.isfile(ccd_file):
            print("ERROR: File 'ZTF_corners.txt' Missing")
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
        survey_end = Time("2018-10-31 00:00:00.000").jd

        inp_df = pd.read_pickle(filename)
        out_df = inp_df.copy()
        out_df['filterid'] = out_df['filterid'].apply(lambda band: dict_filterid[band])
        out_df['skynoise'] = [(10 ** (-0.4 * (limmag - 30))) / 5. for limmag in out_df['limMag']]

        if not clean:
            return out_df
        else:
            out_df = out_df[~out_df['fieldid'].isin([880, 881])]
            out_df = out_df[out_df['jd'] > survey_start][out_df['jd'] < survey_end]
            print ("Survey pointings for All ZTF Programs: {0}".format(len(out_df)))
            print ("Survey pointings for MSIP Programs: {0}".format(len(out_df[out_df['progid'] == 1])))
            return out_df

# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# Class for running the Simulation
# ------------------------------------------------------------------------------------------------------------------- #

class RunSim(Load_ZTFData):

    def __init__(self):
        Load_ZTFData.__init__(self)
        self.days_before_start = 30
        self.days_after_end = 30

        logging.info('Days before Survey Start: {0}'.format(self.days_before_start))
        logging.info('Days after Survey End: {0}'.format(self.days_after_end))

    def make_file_name(self, args, i):
        if not os.path.exists(DIR_OUTPUT):
            os.makedirs(DIR_OUTPUT)

        file_name = "LCS_{0}_{1}_{2}_{3}_{4}.pkl".format(args.type, args.redshift[0], args.redshift[1], args.rate, i)
        return DIR_OUTPUT + file_name

    def get_magnetar_params(self, file_param):
        try:
            prop_df = pd.read_csv(file_param, sep='\s+', comment='#')
        except OSError or FileNotFoundError:
            print("ERROR: File with Magnetar Parameters Missing. Using default parameters instead...")
            SLSN_prop = {'P': [4.78, 2.93, 2.28, 0.98, 3.5],
                         'B': [2.03e14, 1.23e14, 1.8e13, 0.49e14, 1.56e14],
                         'Mej': [2.19, 4.54, 6.27, 33.71, 2.75],
                         'Vej': [5.12, 9.02, 7.01, 8.71, 8.38],
                         'kappa': [0.1, 0.16, 0.16, 0.19, 0.12],
                         'kappa_gamma': [0.06, 0.57, 0.01, 0.01, 0.04],
                         'Mns': [1.85, 1.83, 1.83, 1.8, 1.8],
                         'Tf': [6.58e3, 8e3, 6.48e3, 6.78e3, 5.07e3]}
        else:
            SLSN_prop = {}
            for name in ['P', 'B', 'Mej', 'Vej', 'kappa', 'kappa_gamma', 'Mns', 'Tf']:
                SLSN_prop[name] = prop_df[name].values

        return SLSN_prop

    def run_slsn(self, raw_df, fields, ccds, args, i):
        logging.info('zmin: %f ', args.redshift[0])
        logging.info('zmax:  %f ', args.redshift[1])
        logging.info('transient: %s ', args.type)
        logging.info('rate: %f ', args.rate * 1e-7)

        # Create the Model and combine it with propagation effects
        source = MagnetarSource()
        dust = sncosmo.CCM89Dust()
        model = sncosmo.Model(source=source, effects=[dust], effect_names=['host'], effect_frames=['rest'])

        SLSN_prop = self.get_magnetar_params(file_param)

        # Randomly draw parameters from SLSN properties
        def random_parameters(redshifts, model, r_v=2., ebv_rate=0.11, sig_mag=0.5, cosmo=Planck15, **kwargs):
            idx = np.random.randint(0, len(SLSN_prop['P']), len(redshifts))
            out = {'distance': np.array(cosmo.luminosity_distance(redshifts).value)}
            for key, val in SLSN_prop.items():
                out[key] = np.array(val)[idx]
            return out

        plan = simsurvey.SurveyPlan(time=raw_df['jd'], band=raw_df['filterid'], obs_field=raw_df['fieldid'],
                                    skynoise=raw_df['skynoise'], obs_ccd=raw_df['chid'], comment=raw_df['progid'],
                                    ccds=ccds, fields={k: v for k, v in fields.items()
                                                       if k in ['ra', 'dec', 'field_id', 'width', 'height']})

        mjd_range = (plan.cadence['time'].min() - self.days_before_start,
                     plan.cadence['time'].max() + self.days_after_end)

        tr = simsurvey.get_transient_generator((args.redshift[0], args.redshift[1]),
                                               ratefunc=lambda z: args.rate * 1e-7,
                                               dec_range=(-31, 90),
                                               mjd_range=(mjd_range[0], mjd_range[1]),
                                               sfd98_dir=DIR_SFD,
                                               transientprop={'lcmodel': model, 'lcsimul_func': random_parameters})

        survey = simsurvey.SimulSurvey(generator=tr, plan=plan)
        lcs = survey.get_lightcurves(progress_bar=True, notebook=False)

        output_filename = self.make_file_name(args, i)
        lcs.save(output_filename)
        logging.info('filename %s', output_filename)


def main():
    """
    Step 1: Load all relevant information
    Step 2: Apply selection to ZTF pointings
    """

    logging.basicConfig(filename="SimZTF.log", level=logging.DEBUG)
    logging.info('data file: %s ', datetime.datetime.now())

    parser = ArgumentParser(description='SLSN Rate simulation for ZTF')

    parser.add_argument('-type', type=str, default="SLSN",
                        help='Transient type ("Ia", "IIP", "IIn", "SLSN")')

    parser.add_argument('-z', '--redshift', default=[0, 0.04], nargs=2,
                        help='Redshift Range', type=float)

    parser.add_argument('-data_file', type=str, default=survey_file,
                        help='ZTF Input Filename')

    parser.add_argument('-rate', type=float, default=2.5,
                        help='modify rate parameter for SN in terms of 10^-7 / yr / MPC, default = 2.5 x 10^-7')

    parser.add_argument('-runs', type=int, default=1,
                        help='run same configuration "runs" times')

    args = parser.parse_args()

    run_Sim = RunSim()

    # Load ZTF data
    data_loader = Load_ZTFData()
    data_loader.load_ztf_bands()
    fields, ccds = data_loader.load_fields_ccd(DIR_INPUT)

    df = data_loader.load_modified_input(survey_file)
    logging.info('data file: %s ', survey_file)

    # Run Simulation
    if args.type == "SLSN":
        for i in range(args.runs):
            run_Sim.run_slsn(df, fields, ccds, args, i)
    else:
        print ("Only Meant to run for SLSN Magnetar Model")
        sys.exit(1)


if __name__ == '__main__':
    main()
