# Version 1
# Ludwig Rauch
# Last Modified April 2019


import os
import sys
import pandas as pd
import numpy as np
from astropy.cosmology import Planck15
from astropy.table import Table
from astropy.time import Time
import simsurvey
import sncosmo
import pickle
import datetime
from argparse import ArgumentParser
import logging
from scipy.interpolate import (InterpolatedUnivariateSpline as Spline1d,
                               RectBivariateSpline as Spline2d)

from scipy.integrate import cumtrapz


#Define simulation folder: Telescope parameters
file_path = '/data/lyan/simsurvey/data/'
#Define input file path
in_file_path = '/data/lyan/simsurvey/data/'
#Define output file path
out_file_path = '/data/lyan/simsurvey/output/'
#define folder with dust maps
sfd98_dir = '/data/lyan/simsurvey/data/sfd98/'

#define ponting file:
ztf_pointing_file = "FullAlertQuery_20190408.npy" # raw file from alert query

ztf_cleaned_file = "df_sim_ztf.p" # df of reduced file to speed things up

#helper

def blackbody(wl, T=6e3, R=1., d=1e-5):
    # wl in Ångströms
    # T in Kelvin
    # R in cm
    # d in Mpc
    # output in erg s^-1 cm^-2 Angstrom^-1
    B = 1.19104295262e+27 * wl ** -5 / (np.exp(1.43877735383e+8 / (wl * T)) - 1)
    return B * (R / (d * 3.0857e24)) ** 2

class MagnetarSource(sncosmo.Source):
    """Units:
    distance    -- Mpc
    P           -- ms
    B           -- G
    M_ej        -- M_sun
    v_ej        -- 10^3 km/s
    M_NS        -- M_sun
    kappa       -- cm^2/g
    kappa_gamma -- cm^2/g
    T_f         -- K
    """
    _param_names = ['distance', 'P', 'B', 'M_ej', 'v_ej', 'M_NS', 'kappa', 'kappa_gamma', 'T_f']
    param_names_latex = ['d', 'P', 'B', 'M_{ej}', 'v_{ej}', 'M_{NS}', '\kappa', '\kappa_\gamma', 'T_f']

    def __init__(self, name=None, version=None, minphase=0.01, maxphase=300.,
                 phase_sampling=1000):
        self.name = name
        self.version = version
        self._minphase = minphase
        self._maxphase = maxphase
        self._phase_sampling = phase_sampling

        self._parameters = np.array([1e-5, 0.7, 0.01, 0.1, 14.7, 1.4, 0.05, 0.01, 6e3])

        self._set_L_out()

    # def get_source_parameters(self):
    #     SLSN_prop = {
    #         'P': [4.78, 2.93, 2.28],
    #         'B': [2.03e14, 1.23e14, 1.8e13],
    #         'M_ej': [2.19, 4.54, 6.27],
    #         'v_ej': [5.12, 9.02, 7.01],
    #         'kappa': [0.1, 0.16, 0.16],
    #         'kappa_gamma': [0.06, 0.57, 0.01],
    #         'M_NS': [1.85, 1.83, 1.83],
    #         'T_f': [6.58e3, 8e3, 6.48e3]
    #     }
    #     return SLSN_prop




    def minwave(self):
        return 1e-100

    def maxwave(self):
        return 1e100

    def minphase(self):
        return self._minphase

    def maxphase(self):
        return self._maxphase

    @property
    def _E_mag(self):
        """eq (3) of Nicholl+ 2017"""
        return 2.6e52 * (self._parameters[5] / 1.4) ** 1.5 * self._parameters[1] ** -2

    @property
    def _t_mag(self):
        """eq (4) of Nicholl+ 2017"""
        return (1.3e5 * (self._parameters[5] / 1.4) ** 1.5 * self._parameters[1] ** 2
                * (self._parameters[2] / 1e14) ** -2)

    @property
    def _t_diff(self):
        """eq (6) of Nicholl+ 2017"""
        return (9.84e6 * (self._parameters[6] * self._parameters[3]
                          / self._parameters[4]) ** 0.5)

    @property
    def _A(self):
        """eq (7) Of Nicholl+ 2017"""
        return 4.77e16 * self._parameters[7] * self._parameters[3] / self._parameters[4] ** 2

    @property
    def _is_L_out_set(self):
        return np.all(self._parameters[1:] == self._L_out_parameters)

    def _set_L_out(self):
        """Solving the integral in eq (4) of Nicholl+ 2017
        using scipy.integrate.cumtrapz (cumulative trapezoidal) and
        then interpolate between the steps
        """
        t = np.linspace(self._minphase, self._maxphase, self._phase_sampling) * 8.64e4

        y = (2 * self._E_mag / self._t_mag / (1 + t / self._t_mag) ** 2
             * t / self._t_diff ** 2 * np.exp((t / self._t_diff) ** 2))

        y_int = np.append([0], cumtrapz(y, t))
        L_out_steps = np.exp(-(t / self._t_diff) ** 2) * (1 - np.exp(-self._A * t ** -2)) * y_int

        self._L_out = Spline1d(t / 8.64e4, L_out_steps)
        self._L_out_parameters = self.parameters[1:]

    def _T_from_L(self, phase):
        """Term in eqs (8) and (9) of Nicholl+ 2017 that is compared to T_f"""
        return (self._L_out(phase) / (7.1256e-4 * (self._parameters[4] * phase * 8.64e12) ** 2)) ** 0.25

    def temperature(self, phase):
        """eq (8) of Nicholl+ 2017"""
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
        """eq (9) of Nicholl+ 2017 in cm"""
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
        """Luminosity from Stefan-Boltzman law using eqs (8) and (9) in erg/s"""
        if not self._is_L_out_set:
            self._set_L_out()

        return 7.1256e-4 * self.temperature(phase) ** 4 * self.radius(phase) ** 2

    def _flux(self, phase, wave):
        """"""
        if not self._is_L_out_set:
            self._set_L_out()

        wave = np.array(wave)
        return np.array([blackbody(wave,
                                   T=self.temperature(p_),
                                   R=self.radius(p_),
                                   d=self._parameters[0])
                         for p_ in phase])


class LoadData():
    """
    Load files for simulation
    """

    def __init__(self):
        directory_ZTF_data = os.path.dirname(file_path)
        if not os.path.exists(directory_ZTF_data):
            print("Missing folder with ZTF camera information")
            sys.exit()

        directory_survey_data = os.path.dirname(in_file_path)
        if not os.path.exists(directory_survey_data):
            print("Missing folder with ZTF pointing information")
            sys.exit()
        directory_output_data = os.path.dirname(out_file_path)
        if not os.path.exists(directory_output_data):
            print("No output folder defined. Creating one")
            os.makedirs(directory_output_data)

        if not os.path.exists(sfd98_dir):
            print("No dust file folder defined")
            sys.exit()

    def load_fields_ccd(self,file_path):

        fields_file = file_path + 'ztf_fields.txt'
        if not os.path.isfile(fields_file):
            print("I can not locate ztf_fields.txt file")

        ccd_file = file_path + 'ztf_rc_corners.txt'
        if not os.path.isfile(ccd_file):
            print("I can not locate ztf_rc_corners.txt file")

        fields_raw = np.genfromtxt(fields_file, comments='%')

        fields = {'field_id': np.array(fields_raw[:, 0], dtype=int),
                  'ra': fields_raw[:, 1],
                  'dec': fields_raw[:, 2]}

        # ccd/channel defined here for each quadrant
        ccd_corners = np.genfromtxt(ccd_file, skip_header=1)
        ccds = [ccd_corners[np.array([0, 1, 2, 3]) + 4 * k, :2] for k in range(64)]

        return fields, ccds

    def load_ztf_bands(self, bandpass_dir=file_path):
        bands = {
            'ztfg': 'ztfg_eff.txt',
            'ztfr': 'ztfr_eff.txt',
            'ztfi': 'ztfi_eff.txt',
        }

        for bandname in bands.keys():
            fname = bands[bandname]
            b = np.loadtxt(os.path.join(bandpass_dir, fname))
            band = sncosmo.Bandpass(b[:, 0], b[:, 1], name=bandname)
            sncosmo.registry.register(band, force=True)

    def load_pointings(self, file_path, file):
        file = np.load(file_path+file)
        alert_jd = []
        alert_limMag = []
        alert_chid = []
        alert_progid = []
        alert_fieldid = []
        alert_filterid = []

        for i in range(len(file)):
            alert_jd.append(file[i][0])
            alert_limMag.append(file[i][1])
            alert_chid.append(int(file[i][2]))
            alert_progid.append(file[i][3])
            alert_fieldid.append(int(file[i][4]))
            alert_filterid.append(int(file[i][5]))

        alert_data = {}
        alert_data["jd"] = alert_jd
        alert_data["limMag"] = alert_limMag
        alert_data["chid"] = alert_chid
        alert_data["progid"] = alert_progid
        alert_data["fieldid"] = alert_fieldid
        alert_data["filterid"] = alert_filterid

        return pd.DataFrame(alert_data)

    @staticmethod
    def rename_filter(fid):
        if fid == 1:
            return "ztfg"
        if fid == 2:
            return "ztfr"
        if fid == 3:
            return "ztfi"


    def clean_df(self, df):
        """
        ToDo implement variable cut parameters
        :param df:
        :return:
        """
        lim_mag_cut = 18
        survey_start = Time("2018-06-01 00:00:00.000").jd  # Survey start
        survey_end = Time("2018-12-31 00:00:00.000").jd

        n_start = len(df)

        df['skynoise'] = [10 ** (-0.4 * (i - 30)) / 5. for i in df['limMag']]
        df["filter"] = [LoadData.rename_filter(i) for i in df['filterid']]
        df = df[df["fieldid"] < 1896]
        df = df[df["limMag"] > lim_mag_cut]

        df = df[df["jd"] > survey_start]
        df = df[df["jd"] < survey_end]

        logging.info('Survey start: %f ', survey_start)
        logging.info('Survey end: %f ', survey_end)
        logging.info('Reduced pointings by %f percent', 100 * len(df) / n_start)

        return df

    def get_cleaned_pointings_from_raw(self, in_file_path, ztf_pointing_file):
        raw_data = self.load_pointings(in_file_path, ztf_pointing_file)
        return self.clean_df(raw_data)

    def get_cleaned_saved_df(self, in_file_path, ztf_pointing_file):
        return pickle.load(open(in_file_path+ztf_pointing_file, 'rb'))

class RunSim(LoadData):

    def __init__(self):

        LoadData.__init__(self)
        self.days_prior_start = 100
        self.days_after_end = 30
        # self.zmin = 0
        # self.zmax =0.01
        # self.transient = "IIP"
        # self.template = "nugent"

        logging.info('days prior to survey start: %f ', self.days_prior_start)
        logging.info('days after survey end end: %f ', self.days_after_end)


    def make_folder_structure(self, args):
        """

        :param type: use transient class to create folder structure
        :return: target folder path to save sim output
        """

        target_folder_path = out_file_path  + args.type + "/"
        directory_type_folder = os.path.dirname(target_folder_path)

        if not os.path.exists(directory_type_folder):
            os.makedirs(directory_type_folder)

        return target_folder_path

    def make_file_name(self, args ,i):
        return 'lcs_%s_%s_%s_%s_%s_%s' % (args.type, args.template, args.redshift[0], args.redshift[1], i, args.name)



    def run_default(self, df, fields, ccds, args, i):


        logging.info('zmin: %f ', args.redshift[0])
        logging.info('zmax:  %f ', args.redshift[1])
        logging.info('transient: %s ', args.type)
        logging.info('template: %s ', args.template)
        logging.info('rate: %f ', args.rate * 10**(-5))


        target_folder_path = self.make_folder_structure(args)
        file_name = self.make_file_name(args, i)+".pkl"
        outfile = target_folder_path + file_name

        plan = simsurvey.SurveyPlan(time=df['jd'], band=df['filter'], obs_field=df['fieldid'], obs_ccd=df['chid'],
                                    skynoise=df['skynoise'], comment=df['progid'],
                                    fields={k: v for k, v in fields.items()
                                            if k in ['ra', 'dec', 'field_id',
                                                     'width', 'height']}, ccds=ccds)

        mjd_range = (plan.pointings['time'].min()-self.days_prior_start, plan.pointings['time'].max()
                     + self.days_after_end)

        tr = simsurvey.get_transient_generator((args.redshift[0], args.redshift[1]),
                                               transient=args.type,
                                               ratefunc=lambda z: args.rate * 10**(-5),
                                               template=args.template,
                                               dec_range=(-30, 90),
                                               ra_range=(0, 360),
                                               mjd_range=(mjd_range[0],
                                                          mjd_range[1]),
                                               sfd98_dir=sfd98_dir)

        survey = simsurvey.SimulSurvey(generator=tr, plan=plan)
        lcs = survey.get_lightcurves(progress_bar=True, notebook=False)

        lcs.save(outfile)
        logging.info('filename %s', outfile)
        logging.info('simulated LCs %f ', len([i for i in lcs]))
        return 0

    def run_slsn(self, df, fields, ccds, args, i):


        logging.info('zmin: %f ', args.redshift[0])
        logging.info('zmax:  %f ', args.redshift[1])
        logging.info('transient: %s ', args.type)
        logging.info('template: %s ', args.template)
        logging.info('rate: %f ', args.rate * 10**(-5))


        target_folder_path = self.make_folder_structure(args)
        file_name = self.make_file_name(args, i)+".pkl"
        outfile = target_folder_path + file_name

        source = MagnetarSource()

        # Create the model that combines it with propagation effects
        dust = sncosmo.CCM89Dust()
        model = sncosmo.Model(source=source,
                              effects=[dust],
                              effect_names=['host'],
                              effect_frames=['rest'])

        #SLSN_prop = MagnetarSource.get_source_parameters()
        SLSN_prop = {
            'P': [4.78, 2.93, 2.28, 0.98, 3.5],
            'B': [2.03e14, 1.23e14, 1.8e13, 0.49e14, 1.56e14],
            'M_ej': [2.19, 4.54, 6.27, 33.71, 2.75],
            'v_ej': [5.12, 9.02, 7.01, 8.71, 8.38],
            'kappa': [0.1, 0.16, 0.16, 0.19, 0.12],
            'kappa_gamma': [0.06, 0.57, 0.01, 0.01, 0.04],
            'M_NS': [1.85, 1.83, 1.83, 1.8, 1.8],
            'T_f': [6.58e3, 8e3, 6.48e3, 6.78, 5.07]
        }


        def random_parameters(redshifts, model,
                              r_v=2., ebv_rate=0.11, sig_mag = 0.5,
                              cosmo=Planck15,
                              **kwargs):
            #SLSN_prop = val_SLSN_prop

            #use this to fix amplitude
            out = {'distance': np.array(cosmo.luminosity_distance(redshifts).value)}

            #use this to change amplitude
            #out = {'distance': np.array(cosmo.luminosity_distance(redshifts).value)
            #                   * 10**(-0.4*np.random.normal(0, sig_mag))}

            #use this for ramdom drawing from SLSN properties
            #idx = np.random.randint(0, len(SLSN_prop['P']), len(redshifts))

            #use this to fix a model
            idx = np.random.randint(4, 5, len(redshifts)) #34 for min, 45 for max

            for k, v in SLSN_prop.items():
                out[k] = np.array(v)[idx]

            return out

        #random_parameters = MagnetarSource.random_parameters()

        tr_prop = {
            'lcmodel': model,
            'lcsimul_func': random_parameters
        }

        plan = simsurvey.SurveyPlan(time=df['jd'], band=df['filter'], obs_field=df['fieldid'], obs_ccd=df['chid'],
                                    skynoise=df['skynoise'], comment=df['progid'],
                                    fields={k: v for k, v in fields.items()
                                            if k in ['ra', 'dec', 'field_id',
                                                     'width', 'height']}, ccds=ccds)

        mjd_range = (plan.pointings['time'].min()-self.days_prior_start, plan.pointings['time'].max()
                     + self.days_after_end)


        tr = simsurvey.get_transient_generator((args.redshift[0], args.redshift[1]),
                                               ratefunc=lambda z:  args.rate * 10**(-7),
                                               dec_range=(-30, 90),
                                               mjd_range=(mjd_range[0], mjd_range[1]),
                                               sfd98_dir=sfd98_dir,
                                               transientprop=tr_prop)

        survey = simsurvey.SimulSurvey(generator=tr, plan=plan)

        lcs = survey.get_lightcurves(progress_bar=True, notebook=False)

        lcs.save(outfile)

        logging.info('filename %s', outfile)
        logging.info('simulated LCs %f ', len([i for i in lcs]))

        return 0





def main():
    """
    Step 1: Load all relevant information
    Step 2: Apply selection to ZTF pointings
    """

    logging.basicConfig(filename="SimZTF.log", level=logging.DEBUG)

    logging.info('data file: %s ', datetime.datetime.now())

    parser = ArgumentParser(description='Rate simulation for ZTF')

    parser.add_argument('-raw', type=int, default=0,
                        help='Use raw data file and apply selection (yes = 1, no = 0) , default = yes ')

    parser.add_argument('-type', type=str, default="IIP",
                        help='Transient type ("Ia", "IIP", "IIn", "SLSN")')

    parser.add_argument('-template', type=str, default="1",
                        help='Transient template (e.g. "salt2" or "nugent")')

    parser.add_argument('-z', '--redshift', default=[0, 0.04], nargs=2,
                        help='redshift boundaries', type=float)

    parser.add_argument('-data_file', type=str, default=ztf_cleaned_file,
                        help='filename of input. If raw file from alerts is used then use clean flag')

    parser.add_argument('-rate', type=float, default=2.5,
                        help='modify rate parameter for SN in terms of 10**-5 /yr /MPC. default 2.5 10**-5 '
                             'For SLSN 10**-7' )

    parser.add_argument('-nmc', type=int, default=1,
                        help='run same configuration nmc times')

    parser.add_argument('-name', type=str, default="1",
                        help='pick name to change output file name')

    args = parser.parse_args()


    run_Sim = RunSim()

    raw_mode = False
    if args.raw == 1:
        raw_mode = True

    #lodad data for survey
    data_loader = LoadData()
    data_loader.load_ztf_bands()
    fields, ccds = data_loader.load_fields_ccd(file_path)

    if raw_mode:
        df = data_loader.get_cleaned_pointings_from_raw(in_file_path, ztf_pointing_file)
        logging.info('data file: %s ', ztf_pointing_file)

    else:
        df = data_loader.get_cleaned_saved_df(in_file_path, args.data_file)
        logging.info('data file: %s ', args.data_file)

    # run simulation

    if args.type != "SLSN":
        for i in range(args.nmc):
            run_Sim.run_default(df, fields, ccds, args, i)
    elif args.type == "SLSN":
        for i in range(args.nmc):
            run_Sim.run_slsn(df, fields, ccds, args, i)



if __name__ == '__main__':
    main()
