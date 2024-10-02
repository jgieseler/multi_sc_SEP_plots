import datetime as dt
import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy
from matplotlib.ticker import AutoMinorLocator
# from matplotlib.transforms import blended_transform_factory
from seppy.loader.psp import calc_av_en_flux_PSP_EPIHI, calc_av_en_flux_PSP_EPILO, psp_isois_load, resample_df
from seppy.loader.soho import calc_av_en_flux_ERNE, soho_load
from seppy.loader.stereo import calc_av_en_flux_HET as calc_av_en_flux_ST_HET
from seppy.loader.stereo import calc_av_en_flux_SEPT, stereo_load
from seppy.loader.wind import wind3dp_load
from seppy.util import bepi_sixs_load, calc_av_en_flux_sixs
from solo_epd_loader import combine_channels as calc_av_en_flux_EPD
from solo_epd_loader import epd_load, calc_ept_corrected_e
from sunpy.coordinates import frames, get_horizons_coord
from tqdm import tqdm


# make selections
#############################################################


# first_date = dt.datetime(2021, 4, 17)
# last_date = dt.datetime(2021, 4, 19)
startdate = dt.datetime(2023, 3, 13, 0, 0)
enddate = dt.datetime(2023, 3, 15, 23, 59)
plot_period = '60h'
averaging = '20min'  # '5min'  # None
averaging2 = '30min'  # '5min'  # None

lower_proton = False  # True if 13 MeV protons should be used instead of 25+ MeV


Bepi = True
Maven = True
PSP = True
SOHO = True
SOLO = True
STEREO = True
WIND = True


# SOHO:
erne = True
erne_maven = False
ephin_p = False  # not included yet! All proton data is set to -9e9 during loading bc. it's not fully implemented yet
ephin_e = True  # not included yet!

# SOLO:
ept = True
het = True
het_maven = False
# ept_use_corr_e = False  # not included yet!

# STEREO:
sept_e = True
sept_p = True
stereo_het = True
sta_het_maven = False
let = False

wind3dp_p = True
wind3dp_e = True
#############################################################

# omit some warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=sunpy.util.SunpyUserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# manually define seaborn-colorblind colors
seaborn_colorblind = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']  # blue, green, orange, magenta, yello, light blue
# change some matplotlib plotting settings
SIZE = 22
plt.rc('font', size=SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 3.0


# some plot options
intensity_label = 'Flux\n/(s cm² sr MeV)'
linewidth = 3  # 1.5
outpath = None  # os.getcwd()
plot_e_100 = False
plot_e_1 = False
plot_p = True
plot_p_maven = True
save_fig = True
outpath = 'plots/'  # '/Users/dresing/Documents/Proposals/SERPENTINE_H2020/Cycle25_Multi-SC_SEP_Event_List/Multi_sc_plots'

# dates = pd.date_range(start=first_date, end=last_date, freq=plot_period)
# startdate = dates[i].to_pydatetime()
# startdate = dates[i].to_pydatetime()
# enddate = startdate + pd.Timedelta(plot_period)
outfile = f'{outpath}{os.sep}Multi_sc_plot_{startdate.date()}_{plot_period}_{averaging}-av.png'

if Bepi:
    # av_bepi = 10
    sixs_resample = averaging2  # '10min'
    sixs_ch_e1 = 5  # [5, 6]  # => 5
    sixs_ch_e100 = 2
    sixs_ch_p = 8  # [8, 9]  # we want 'P8'-'P9' averaged
    if lower_proton:
        sixs_ch_p = [7]
    sixs_ch_p_maven = 6  # 7
    sixs_side = 2
    sixs_color = 'orange'  # seaborn_colorblind[4]  # orange?
    # sixs_path = '/home/gieseler/uni/bepi/data/bc_mpo_sixs/data_csv/cruise/sixs-p/raw'
    sixs_path = '/Users/jagies/data/bepi/bc_mpo_sixs/data_csv/cruise/sixs-p/raw'
if Maven:
    maven_resample = averaging  # '10min'
    # maven_ch_e1 = 0
    maven_ch_e100 = 9  # 11
    # if higher_e100:
    #     maven_ch_e100 = 4
    maven_ch_p = 27  #0
    if lower_proton:
        maven_ch_p = 27  # 26
    maven_ch_p_new_1 = 0
    maven_ch_p_new_2 = 0
    maven_color = 'chocolate'  # seaborn_colorblind[4]  # orange?
    maven_path = 'plots/march_13/'
    maven_efname = 'Mar2023NDresing_mvn_5min_SEP1F_elec_EFLUX_open.txt'
    maven_ifname = 'Mar2023NDresing_mvn_5min_SEP1F_ion_EFLUX_open.txt'
    maven_ifname_new_1 = 'Mar2023NDresing_mdap_dint_md1_sta.txt'
    maven_ifname_new_2 = 'Mar2023NDresing_mdap_dint_md1sub_sta.txt'
    maven_time_format = '%Y-%m-%d/%H:%M:%S.%f'
    maven_e_channels = [21.0497, 22.4964, 23.9430, 26.1130, 29.0062, 32.6228, 37.6860, 44.1959, 52.8757, 64.4487, 79.6384, 99.8912, 126.654, 161.373, 206.942]
    maven_p_channels = [20.8326, 22.2503, 23.6680, 25.7945, 28.6298, 32.1740, 37.1359, 43.5155, 52.0215, 63.3630, 78.2486, 98.0961, 124.323, 158.347, 203.004,
                        261.838, 339.810, 442.592, 577.272, 755.190, 989.816, 1298.87, 1706.45, 2243.04, 2949.05, 3879.05, 5111.72, 6687.22]
    maven_p_channels_new_1 = ['23 - 27.4 MeV']
    maven_p_channels_new_2 = ['13.0 - 17.3 MeV', '18.3 - 24.4 MeV', '25.9 - 34.5 MeV', '36.5 - 48.7 MeV', '51.6 - 68.8 MeV', '72.9 - 102.9 MeV']

if SOHO:
    soho_ephin_color = 'k'
    soho_erne_color = 'k'  # seaborn_colorblind[5]  # 'green'
    # av_soho = av
    soho_erne_resample = averaging  # '30min'
    soho_ephin_resample = averaging  # '30min'
    # soho_path = '/home/gieseler/uni/soho/data/'
    soho_path = '/Users/jagies/data/soho/'
    if erne:
        erne_p_ch = [2, 3]  # [3, 4]  # [0]  # [4,5]  # 2
        if lower_proton:
            erne_p_ch = [0]
        erne_p_ch_maven = 0
    if ephin_e:
        ephin_ch_e1 = 'E150'  # 'E1300'
        # ephin_e_intercal = 1/14.
    # if ephin_p:
    #     ephin_ch_p = 'p25'
if SOLO:
    solo_ept_color = seaborn_colorblind[5]  # 'blue'
    solo_het_color = seaborn_colorblind[5]  # seaborn_colorblind[0]  # 'blue' # seaborn_colorblind[1]
    sector = 'sun'
    ept_ch_e100 = [14, 18]  # [25]
    het_ch_e1 = [0]  # [0, 1]  # cf. het_energies
    ept_ch_p = 63  # [50, 56]  # 50-56
    het_ch_p = [18, 19]  # [19, 24]  # [18, 19]  # cf. het_energies
    het_ch_p_maven = [11, 13]
    if lower_proton:
        het_ch_p = [11, 12]
    solo_ept_resample = averaging2
    solo_het_resample = averaging2
    # solo_path = '/home/gieseler/uni/solo/data/'
    solo_path = '/Users/jagies/data/solo/'
if STEREO:
    stereo_sept_color = 'orangered'  # seaborn_colorblind[3]  #
    stereo_het_color = 'orangered'  # seaborn_colorblind[3]  # 'coral'
    stereo_let_color = 'orangered'  # seaborn_colorblind[3]  # 'coral'
    sector = 'sun'
    sept_ch_e100 = [6, 7]  # [12, 16]
    sept_ch_p = 31  # [25, 30]
    st_het_ch_e = 0  # [0, 1]   cf. sta_het_meta['channels_dict_df_e']
    st_het_ch_p = [4]  # [5, 8]  # 3  #7 #3   cf. sta_het_meta['channels_dict_df_p']
    st_het_ch_p_maven = [0, 1]
    if lower_proton:
        st_het_ch_p = [0]
    let_ch = 5  # 1
    sta_het_resample = averaging
    sta_sept_resample = averaging
    sta_let_resample = averaging
    # stereo_path = '/home/gieseler/uni/stereo/data/'
    stereo_path = '/Users/jagies/data/stereo/'
if WIND:
    wind_color = 'k'  # 'dimgrey'
    wind3dp_ch_e100 = 3
    wind3dp_ch_p = 8  # 6
    wind_3dp_resample = averaging  # '30min'
    wind_3dp_threshold = None  # 1e3/1e6  # None
    # wind_path = '/home/gieseler/uni/wind/data/'
    wind_path = '/Users/jagies/data/wind/'
if PSP:
    psp_epilo_ch_e100 = [4, 5]  # cf. psp_epilo_energies
    psp_het_ch_e = [4, 5]  # [3, 10]  # cf. psp_het_energies
    psp_het_ch_p = [7]  # [8, 9]  # cf. psp_het_energies
    psp_het_ch_p_maven = [1]  # [4]
    if lower_proton:
        psp_het_ch_p = [4]
    psp_epilo_channel = 'F'
    psp_epilo_viewing = 3  # 3="sun", 7="antisun"
    psp_epilo_threshold = None  # 1e2  # None
    # psp_path = '/home/gieseler/uni/psp/data/'
    psp_path = '/Users/jagies/data/psp/'
    psp_het_resample = averaging
    psp_epilo_resample = averaging
    psp_het_color = 'blueviolet'

# LOAD DATA
##################################################################

if WIND:
    if wind3dp_e:
        print('loading wind/3dp e')
        wind3dp_e_df, wind3dp_e_meta = wind3dp_load(dataset="WI_SFSP_3DP",
                                                    startdate=startdate,
                                                    enddate=enddate,
                                                    resample=wind_3dp_resample,
                                                    threshold=wind_3dp_threshold,
                                                    multi_index=False,
                                                    path=wind_path,
                                                    max_conn=1)
        wind3dp_ch_e = wind3dp_ch_e100

    if wind3dp_p:
        print('loading wind/3dp p')
        wind3dp_p_df, wind3dp_p_meta = wind3dp_load(dataset="WI_SOSP_3DP", startdate=startdate, enddate=enddate, resample=wind_3dp_resample, multi_index=False, path=wind_path, max_conn=1)

if STEREO:
    if stereo_het:
        print('loading stereo/het')
        sta_het_e_labels = ['0.7-1.4 MeV', '1.4-2.8 MeV', '2.8-4.0 MeV']
        sta_het_p_labels = ['13.6-15.1 MeV', '14.9-17.1 MeV', '17.0-19.3 MeV', '20.8-23.8 MeV', '23.8-26.4 MeV', '26.3-29.7 MeV', '29.5-33.4 MeV', '33.4-35.8 MeV', '35.5-40.5 MeV', '40.0-60.0 MeV']

        sta_het_df, sta_het_meta = stereo_load(instrument='het', startdate=startdate, enddate=enddate, spacecraft='sta', resample=sta_het_resample, path=stereo_path, max_conn=1)

    if let:
        print('loading stereo/let')
        # for H and He4:
        let_chstring = ['1.8-2.2 MeV', '2.2-2.7 MeV', '2.7-3.2 MeV', '3.2-3.6 MeV', '3.6-4.0 MeV', '4.0-4.5 MeV', '4.5-5.0 MeV', '5.0-6.0 MeV', '6.0-8.0 MeV', '8.0-10.0 MeV', '10.0-12.0 MeV', '12.0-15.0 MeV']

        sta_let_df, sta_let_meta = stereo_load(instrument='let', startdate=startdate, enddate=enddate, spacecraft='sta', resample=sta_let_resample, path=stereo_path, max_conn=1)
    if sept_e:
        print('loading stereo/sept e')

        sta_sept_df_e, sta_sept_dict_e = stereo_load(instrument='sept', startdate=startdate, enddate=enddate, spacecraft='sta', sept_species='e', sept_viewing=sector, resample=sta_sept_resample, path=stereo_path, max_conn=1)
        sept_ch_e = sept_ch_e100

    if sept_p:
        print('loading stereo/sept p')

        sta_sept_df_p, sta_sept_dict_p = stereo_load(instrument='sept', startdate=startdate, enddate=enddate, spacecraft='sta', sept_species='p', sept_viewing=sector, resample=sta_sept_resample, path=stereo_path, max_conn=1)

    sectors = {'sun': 0, 'asun': 1, 'north': 2, 'south': 3}
    sector_num = sectors[sector]

if SOHO:
    if ephin_e or ephin_p:
        print('loading soho/ephin')
        # ephin = eph_rl2_loader(startdate.year, startdate.timetuple().tm_yday, doy2=enddate.timetuple().tm_yday, av=av_soho)
        soho_ephin, ephin_energies = soho_load(dataset="SOHO_COSTEP-EPHIN_L2-1MIN",
                                               startdate=startdate,
                                               enddate=enddate,
                                               path=soho_path,
                                               resample=soho_ephin_resample,
                                               pos_timestamp='center')

    if erne:
        print('loading soho/erne')
        erne_chstring = ['13-16 MeV', '16-20 MeV', '20-25 MeV', '25-32 MeV', '32-40 MeV', '40-50 MeV', '50-64 MeV', '64-80 MeV', '80-100 MeV', '100-130 MeV']
        # soho_p = ERNE_HED_loader(startdate.year, startdate.timetuple().tm_yday, doy2=enddate.timetuple().tm_yday, av=av_soho)
        soho_erne, erne_energies = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN", startdate=startdate, enddate=enddate, path=soho_path, resample=soho_erne_resample, max_conn=1)

if PSP:
    print('loading PSP/EPIHI-HET data')
    psp_het, psp_het_energies = psp_isois_load('PSP_ISOIS-EPIHI_L2-HET-RATES60', startdate, enddate, path=psp_path, resample=None)
    # psp_let1, psp_let1_energies = psp_isois_load('PSP_ISOIS-EPIHI_L2-LET1-RATES60', startdate, enddate, path=psp_path, resample=psp_resample)
    if len(psp_het) == 0:
        print(f'No PSP/EPIHI-HET 60s data found for {startdate.date()} - {enddate.date()}. Trying 3600s data.')
        psp_het, psp_het_energies = psp_isois_load('PSP_ISOIS-EPIHI_L2-HET-RATES3600', startdate, enddate, path=psp_path, resample=None)
        psp_3600 = True
        psp_het_resample = None

    if plot_e_100:
        print('loading PSP/EPILO PE data')
        psp_epilo, psp_epilo_energies = psp_isois_load('PSP_ISOIS-EPILO_L2-PE',
                                                       startdate, enddate,
                                                       epilo_channel=psp_epilo_channel,
                                                       epilo_threshold=psp_epilo_threshold,
                                                       path=psp_path, resample=None)
        if len(psp_epilo) == 0:
            print(f'No PSP/EPILO PE data for {startdate.date()} - {enddate.date()}')

if SOLO:
    data_product = 'l2'
    sdate = startdate
    edate = enddate
    if ept:
        if plot_e_100 or plot_p:
            print('loading solo/ept e & p')
            try:
                ept_p, ept_e, ept_energies = epd_load(sensor='EPT', viewing=sector, level=data_product, startdate=sdate, enddate=edate, path=solo_path, autodownload=True)
            except(Exception):
                print(f'No SOLO/EPT data for {startdate.date()} - {enddate.date()}')
                ept_e = []
    if het:
        if plot_e_1 or plot_p:
            print('loading solo/het e & p')
            try:
                het_p, het_e, het_energies = epd_load(sensor='HET', viewing=sector, level=data_product, startdate=sdate, enddate=edate, path=solo_path, autodownload=True)
            except(Exception):
                print(f'No SOLO/HET data for {startdate.date()} - {enddate.date()}')
                het_e = []
                het_p = []

if Bepi:
    print('loading Bepi/SIXS')
    # sixs_e, sixs_chstrings = bepi_sixs_loader(startdate.year, startdate.month, startdate.day, sixs_side_e, av=sixs_resample)
    # sixs_p, sixs_chstrings = bepi_sixs_loader(startdate.year, startdate.month, startdate.day, sixs_side_p, av=sixs_resample)
    # sixs_ch_e = sixs_ch_e100
    sixs_df, sixs_meta = bepi_sixs_load(startdate=startdate,
                                        enddate=enddate,
                                        side=sixs_side,
                                        path=sixs_path)
    if len(sixs_df) > 0:
        sixs_df_p = sixs_df[[f"P{i}" for i in range(1, 10)]]
        sixs_df_e = sixs_df[[f"E{i}" for i in range(1, 8)]]

if Maven:
    print('loading Maven')
    maven_e = pd.read_csv(maven_path+maven_efname, sep=r"\s+",
                          names=['Time', 'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14'])
    maven_e.index = pd.to_datetime(maven_e['Time'], errors='coerce')
    maven_e.drop('Time', axis=1, inplace=True)

    maven_p = pd.read_csv(maven_path+maven_ifname, sep=r"\s+",
                          names=['Time', 'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15',
                                 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27'])
    maven_p.index = pd.to_datetime(maven_p['Time'], errors='coerce')
    maven_p.drop('Time', axis=1, inplace=True)

    maven_p_new_1 = pd.read_csv(maven_path+maven_ifname_new_1, sep=r"\s+", names=['Time', 'E0'])
    maven_p_new_1.index = pd.to_datetime(maven_p_new_1['Time'], errors='coerce')
    maven_p_new_1.drop('Time', axis=1, inplace=True)

    maven_p_new_2 = pd.read_csv(maven_path+maven_ifname_new_2, sep=r"\s+", names=['Time', 'E0', 'E1', 'E2', 'E3', 'E4', 'E5'])
    maven_p_new_2.index = pd.to_datetime(maven_p_new_2['Time'], errors='coerce')
    maven_p_new_2.drop('Time', axis=1, inplace=True)

""" AVERAGE ENERGY CHANNELS """
####################################################
if SOLO:
    if ept:
        if len(ept_e) > 0:
            if plot_e_100:
                df_ept_e = ept_e['Electron_Flux']
                ept_en_str_e = ept_energies['Electron_Bins_Text'][:]

                # if ept_use_corr_e:
                #     print('correcting e')
                #     ion_cont_corr_matrix = np.loadtxt('EPT_ion_contamination_flux_paco.dat')
                #     Electron_Flux_cont = np.zeros(np.shape(df_ept_e))
                #     for tt in range(len(df_ept_e)):
                #         Electron_Flux_cont[tt, :] = np.matmul(ion_cont_corr_matrix, df_ept_p.values[tt, :])
                #     df_ept_e = df_ept_e - Electron_Flux_cont

                df_ept_e, ept_chstring_e = calc_av_en_flux_EPD(ept_e, ept_energies, ept_ch_e100, 'ept')

                if isinstance(solo_ept_resample, str):
                    df_ept_e = resample_df(df_ept_e, solo_ept_resample)
            if plot_p:
                df_ept_p = ept_p['Ion_Flux']
                ept_en_str_p = ept_energies['Ion_Bins_Text'][:]
                df_ept_p, ept_chstring_p = calc_av_en_flux_EPD(ept_p, ept_energies, ept_ch_p, 'ept')
                if isinstance(solo_ept_resample, str):
                    df_ept_p = resample_df(df_ept_p, solo_ept_resample)

    if het:
        if len(het_e) > 0:
            if plot_e_1:
                print('calc_av_en_flux_HET e')
                df_het_e, het_chstring_e = calc_av_en_flux_EPD(het_e, het_energies, het_ch_e1, 'het')
                if isinstance(solo_het_resample, str):
                    df_het_e = resample_df(df_het_e, solo_het_resample)
            if plot_p:
                print('calc_av_en_flux_HET p')
                df_het_p, het_chstring_p = calc_av_en_flux_EPD(het_p, het_energies, het_ch_p, 'het')
                if isinstance(solo_het_resample, str):
                    df_het_p = resample_df(df_het_p, solo_het_resample)
                df_het_p_maven, het_chstring_p_maven = calc_av_en_flux_EPD(het_p, het_energies, het_ch_p_maven, 'het')
                if isinstance(solo_het_resample, str):
                    df_het_p_maven = resample_df(df_het_p_maven, solo_het_resample)

if STEREO:
    if sept_e:
        if type(sept_ch_e) == list and len(sta_sept_df_e) > 0:
            sta_sept_avg_e, sept_chstring_e = calc_av_en_flux_SEPT(sta_sept_df_e, sta_sept_dict_e, sept_ch_e)
        else:
            sta_sept_avg_e = []
            sept_chstring_e = ''

    if sept_p:
        if type(sept_ch_p) == list and len(sta_sept_df_p) > 0:
            sta_sept_avg_p, sept_chstring_p = calc_av_en_flux_SEPT(sta_sept_df_p, sta_sept_dict_p, sept_ch_p)
        else:
            sta_sept_avg_p = []
            sept_chstring_p = ''

    if stereo_het:
        if type(st_het_ch_e) == int:
            st_het_ch_e = [st_het_ch_e]
        if type(st_het_ch_e) == list and len(sta_het_df) > 0:
            sta_het_avg_e, st_het_chstring_e = calc_av_en_flux_ST_HET(sta_het_df.filter(like='Electron'),
                                                                      sta_het_meta['channels_dict_df_e'],
                                                                      st_het_ch_e, species='e')
        else:
            sta_het_avg_e = []
            st_het_chstring_e = ''
        if type(st_het_ch_p) == int:
            st_het_ch_p = [st_het_ch_p]
        if type(st_het_ch_p) == list and len(sta_het_df) > 0:
            sta_het_avg_p, st_het_chstring_p = calc_av_en_flux_ST_HET(sta_het_df.filter(like='Proton'),
                                                                      sta_het_meta['channels_dict_df_p'],
                                                                      st_het_ch_p, species='p')
        else:
            sta_het_avg_p = []
            st_het_chstring_p = ''
        if type(st_het_ch_p_maven) == list and len(sta_het_df) > 0:
            sta_het_avg_p_maven, st_het_chstring_p_maven = calc_av_en_flux_ST_HET(sta_het_df.filter(like='Proton'),
                                                                      sta_het_meta['channels_dict_df_p'],
                                                                      st_het_ch_p_maven, species='p')
if SOHO:
    if erne:
        if type(erne_p_ch) == list and len(soho_erne) > 0:
            soho_erne_avg_p, soho_erne_chstring_p = calc_av_en_flux_ERNE(soho_erne.filter(like='PH_'),
                                                                         erne_energies['channels_dict_df_p'],
                                                                         erne_p_ch,
                                                                         species='p',
                                                                         sensor='HET')                                                    
if Bepi:
    if len(sixs_df) > 0:
        # 1 MeV electrons:
        if type(sixs_ch_e1) == list:
            if len(sixs_ch_e1) == 2:
                sixs_df_e1, sixs_e1_en_channel_string = calc_av_en_flux_sixs(sixs_df_e, sixs_ch_e1, 'e')
            if len(sixs_ch_e1) == 1:
                sixs_ch_e1 = sixs_ch_e1[0]
        if type(sixs_ch_e1) == int:
            sixs_df_e1 = sixs_df_e[f'E{sixs_ch_e1}']
            sixs_e1_en_channel_string = sixs_meta['Energy_Bin_str'][f'E{sixs_ch_e1}']
        # >25 MeV protons:
        if type(sixs_ch_p) == list:
            if len(sixs_ch_p) == 2:
                sixs_df_p25, sixs_p25_en_channel_string = calc_av_en_flux_sixs(sixs_df_p, sixs_ch_p, 'p')
            if len(sixs_ch_p) == 1:
                sixs_ch_p = sixs_ch_p[0]
        if type(sixs_ch_p) == int:
            sixs_df_p25 = sixs_df_p[f'P{sixs_ch_p}']
            sixs_p25_en_channel_string = sixs_meta['Energy_Bin_str'][f'P{sixs_ch_p}']
        if type(sixs_ch_p_maven) == int:
            sixs_df_p_maven = sixs_df_p[f'P{sixs_ch_p_maven}']
            sixs_p_maven_en_channel_string = sixs_meta['Energy_Bin_str'][f'P{sixs_ch_p_maven}']
        # 100 keV electrons withouth averaging:
        sixs_df_e100 = sixs_df_e[f'E{sixs_ch_e100}']
        sixs_e100_en_channel_string = sixs_meta['Energy_Bin_str'][f'E{sixs_ch_e100}']

        if isinstance(sixs_resample, str):
            sixs_df_e100 = resample_df(sixs_df_e100, sixs_resample)
            sixs_df_e1 = resample_df(sixs_df_e1, sixs_resample)
            sixs_df_p25 = resample_df(sixs_df_p25, sixs_resample)
            sixs_df_p_maven = resample_df(sixs_df_p_maven, sixs_resample)

if Maven:
    if isinstance(maven_resample, str):
        # maven_e = maven_e.resample(maven_resample,label='left').mean()
        # maven_e.index = maven_e.index + pd.tseries.frequencies.to_offset(pd.Timedelta(maven_resample)/2)
        maven_e = resample_df(maven_e, maven_resample)
        maven_p = resample_df(maven_p, maven_resample)
        maven_p_new_1 = resample_df(maven_p_new_1, maven_resample)
        maven_p_new_2 = resample_df(maven_p_new_2, maven_resample)

if PSP:
    if plot_e_1:
        if len(psp_het) > 0:
            print('calc_av_en_flux_PSP_EPIHI e 1 MeV')
            if type(psp_het_ch_e) == int:
                psp_het_ch_e = [psp_het_ch_e]
            df_psp_het_e, psp_het_chstring_e = calc_av_en_flux_PSP_EPIHI(psp_het, psp_het_energies, psp_het_ch_e, 'e', 'het', 'A')
            if isinstance(psp_het_resample, str):
                df_psp_het_e = resample_df(df_psp_het_e, psp_het_resample)
    if plot_p:
        print('calc_av_en_flux_PSP_EPIHI p')
        if type(psp_het_ch_p) == int:
            psp_het_ch_p = [psp_het_ch_p]
        df_psp_het_p, psp_het_chstring_p = calc_av_en_flux_PSP_EPIHI(psp_het, psp_het_energies, psp_het_ch_p, 'p', 'het', 'A')
        if isinstance(psp_het_resample, str):
            df_psp_het_p = resample_df(df_psp_het_p, psp_het_resample)
    if plot_p_maven:
        print('calc_av_en_flux_PSP_EPIHI p_maven')
        if type(psp_het_ch_p_maven) == int:
            psp_het_ch_p_maven = [psp_het_ch_p_maven]
        df_psp_het_p_maven, psp_het_chstring_p_maven = calc_av_en_flux_PSP_EPIHI(psp_het, psp_het_energies, psp_het_ch_p_maven, 'p', 'het', 'A')
        if isinstance(psp_het_resample, str):
            df_psp_het_p_maven = resample_df(df_psp_het_p_maven, psp_het_resample)
    if plot_e_100:
        if len(psp_epilo) > 0:
            print('calc_av_en_flux_PSP_EPILO e 100 keV')
            df_psp_epilo_e, psp_epilo_chstring_e = calc_av_en_flux_PSP_EPILO(psp_epilo,
                                                                             psp_epilo_energies,
                                                                             psp_epilo_ch_e100,
                                                                             species='e',
                                                                             mode='pe',
                                                                             chan=psp_epilo_channel,
                                                                             viewing=psp_epilo_viewing)

            # select energy channel
            # TODO: introduce calc_av_en_flux_PSP_EPILO(). ATM, if list of channels, only first one is selected
            # if type(psp_epilo_ch_e100) is list:
            #    psp_epilo_ch_e100 = psp_epilo_ch_e100[0]
            # df_psp_epilo_e = df_psp_epilo_e.filter(like=f'_E{psp_epilo_ch_e100}_')

            # energy = en_dict['Electron_ChanF_Energy'].filter(like=f'_E{en_channel}_P{viewing}').values[0]
            # energy_low = energy - en_dict['Electron_ChanF_Energy_DELTAMINUS'].filter(like=f'_E{en_channel}_P{viewing}').values[0]
            # energy_high = energy + en_dict['Electron_ChanF_Energy_DELTAPLUS'].filter(like=f'_E{en_channel}_P{viewing}').values[0]
            # chstring_e = np.round(energy_low,1).astype(str) + ' - ' + np.round(energy_high,1).astype(str) + ' keV'

            if isinstance(psp_epilo_resample, str):
                df_psp_epilo_e = resample_df(df_psp_epilo_e, psp_epilo_resample)

##########################################################################################


panels = 0
if plot_e_1:
    panels = panels + 1
if plot_e_100:
    panels = panels + 1
if plot_p:
    panels = panels + 1
if plot_p_maven:
    panels = panels + 1
# fig, axes = plt.subplots(panels, figsize=(24, 15), dpi=200, sharex=True)
fig, axes = plt.subplots(panels, figsize=(20, 13), dpi=200, sharex=True)
axnum = 0
# Intensities
####################################################################
# 100 KEV ELECTRONS
#################################################################
if plot_e_100:
    if panels == 1:
        ax = axes
    else:
        ax = axes[axnum]
    species_string = 'Electrons'
    # if ept_use_corr_e:
    #     species_string = 'Electrons (corrected)'

    if PSP:
        if len(psp_epilo) > 0:
            ax.plot(df_psp_epilo_e.index, df_psp_epilo_e*100, color=psp_het_color, linewidth=linewidth,
                    label='PSP '+r"$\bf{(count\ rate\ *100)}$"+'\nISOIS-EPILO '+psp_epilo_chstring_e+f'\nF (W{psp_epilo_viewing})',
                    drawstyle='steps-mid')

    if Bepi:
        # ax.plot(sixs_e.index, sixs_e[sixs_ch_e], color='orange', linewidth=linewidth, label='BepiColombo\nSIXS '+sixs_chstrings[sixs_ch_e]+f'\nside {sixs_side_e}', drawstyle='steps-mid')
        if len(sixs_df) > 0:
            ax.plot(sixs_df_e100.index, sixs_df_e100, color=sixs_color, linewidth=linewidth, label='BepiColombo/SIXS '+sixs_e100_en_channel_string+f' side {sixs_side}', drawstyle='steps-mid')
    if SOLO:
        if ept and (len(ept_e) > 0):
            flux_ept = df_ept_e.values
            try:
                for ch in ept_ch_e100:
                    ax.plot(df_ept_e.index.values, flux_ept[:, ch], linewidth=linewidth, color=solo_ept_color, label='SOLO\nEPT '+ept_en_str_e[ch, 0]+f'\n{sector}', drawstyle='steps-mid')
            except IndexError:
                ax.plot(df_ept_e.index.values, flux_ept, linewidth=linewidth, color=solo_ept_color, label='SOLO\nEPT '+ept_chstring_e+f'\n{sector}', drawstyle='steps-mid')
    if STEREO:
        if sept_e:
            if type(sept_ch_e) == list and len(sta_sept_avg_e) > 0:
                ax.plot(sta_sept_avg_e.index, sta_sept_avg_e, color=stereo_sept_color, linewidth=linewidth,
                        label='STEREO/SEPT '+sept_chstring_e+f' {sector}', drawstyle='steps-mid')
            elif type(sept_ch_e) == int:
                ax.plot(sta_sept_df_e.index, sta_sept_df_e[f'ch_{sept_ch_e}'], color=stereo_sept_color,
                        linewidth=linewidth, label='STEREO/SEPT '+sta_sept_dict_e.loc[sept_ch_e]['ch_strings']+f' {sector}', drawstyle='steps-mid')

    # if SOHO:
        # if ephin_e:
        #     ax.plot(ephin['date'], ephin[ephin_ch_e][0]*ephin_e_intercal, color=soho_ephin_color,
        #             linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_e][1]+f'/{ephin_e_intercal}',
        #             drawstyle='steps-mid')
    if WIND:
        if len(wind3dp_e_df) > 0:
            # multiply by 1e6 to get per MeV
            ax.plot(wind3dp_e_df.index, wind3dp_e_df[f'FLUX_{wind3dp_ch_e}']*1e6, color=wind_color, linewidth=linewidth, label='Wind/3DP '+wind3dp_e_meta['channels_dict_df']['Bins_Text'][wind3dp_ch_e], drawstyle='steps-mid')

    # ax.set_ylim(7.9e-3, 4.7e1)
    # ax.set_ylim(0.3842003987966555, 6333.090511873226)
    ax.set_yscale('log')
    ax.set_ylabel(intensity_label)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='100 keV '+species_string)
    axnum = axnum + 1

# 1 MEV ELECTRONS
#################################################################
if plot_e_1:
    if panels == 1:
        ax = axes
    else:
        ax = axes[axnum]
        species_string = 'Electrons'
    # if ept_use_corr_e:
    #     species_string = 'Electrons (corrected)'

    if PSP:
        if len(psp_het) > 0:
            # ax.plot(psp_het.index, psp_het[f'A_Electrons_Rate_{psp_het_ch_e}'], color=psp_het_color, linewidth=linewidth,
            #         label='PSP '+r"$\bf{(count\ rates)}$"+'\nISOIS-EPIHI-HET '+psp_het_energies['Electrons_ENERGY_LABL'][psp_het_ch_e][0].replace(' ', '').replace('-', ' - ').replace('MeV', ' MeV')+'\nA (sun)',
            #         drawstyle='steps-mid')
            ax.plot(df_psp_het_e.index, df_psp_het_e*10, color=psp_het_color, linewidth=linewidth,
                    label='PSP '+r"$\bf{(count\ rate\ *10)}$"+'\nISOIS-EPIHI '+psp_het_chstring_e+'\nA (sun)',
                    drawstyle='steps-mid')
    if Bepi:
        # ax.plot(sixs_e.index, sixs_e[sixs_ch_e100], color='orange', linewidth=linewidth,
        #         label='Bepi/SIXS '+sixs_chstrings[sixs_ch_e100]+f' side {sixs_side_e}', drawstyle='steps-mid')
        if len(sixs_df) > 0:
            ax.plot(sixs_df_e1.index, sixs_df_e1, color=sixs_color, linewidth=linewidth,
                    label='BepiColombo\nSIXS '+sixs_e1_en_channel_string+f'\nside {sixs_side}', drawstyle='steps-mid')
    if SOLO:
        if het and (len(het_e) > 0):
            ax.plot(df_het_e.index.values, df_het_e.flux, linewidth=linewidth, color=solo_het_color, label='SOLO\nHET '+het_chstring_e+f'\n{sector}', drawstyle='steps-mid')
    if STEREO:
        if stereo_het:
            if len(sta_het_avg_e) > 0:
                ax.plot(sta_het_avg_e.index, sta_het_avg_e, color=stereo_het_color, linewidth=linewidth,
                        label='STEREO\nHET '+st_het_chstring_e, drawstyle='steps-mid')
    if SOHO:
        if ephin_e:
            # ax.plot(ephin['date'], ephin[ephin_ch_e][0]*ephin_e_intercal, color=soho_ephin_color, linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_e][1]+f'/{ephin_e_intercal}', drawstyle='steps-mid')
            if len(soho_ephin) > 0:
                ax.plot(soho_ephin.index, soho_ephin[ephin_ch_e1]/14, color=soho_ephin_color, linewidth=linewidth, label='SOHO '+r"$\bf{(flux\, /\, 14)}$"+'\nEPHIN '+ephin_energies[ephin_ch_e1], drawstyle='steps-mid', zorder=-99)

    # ax.set_ylim(7.9e-3, 4.7e1)
    # ax.set_ylim(0.3842003987966555, 6333.090511873226)
    ax.set_yscale('log')
    ax.set_ylabel(intensity_label)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='1 MeV '+species_string)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=species_string)
    axnum = axnum + 1

# PROTONS
#################################################################
if plot_p:
    if panels == 1:
        ax = axes
    else:
        ax = axes[axnum]
    if PSP:
        if len(psp_het) > 0:
            # ax.plot(psp_het.index, psp_het[f'A_H_Flux_{psp_het_ch_p}'], color=psp_het_color, linewidth=linewidth,
            #         label='PSP '+r"$\bf{(count\ rates)}$"+'\nISOIS-EPIHI-HET '+psp_het_energies['H_ENERGY_LABL'][psp_het_ch_p][0].replace(' ', '').replace('-', ' - ').replace('MeV', ' MeV')+'\nA (sun)',
            #         drawstyle='steps-mid')
            ax.plot(df_psp_het_p.index, df_psp_het_p, color=psp_het_color, linewidth=linewidth,
                    label='PSP '+'\nISOIS-EPIHI '+psp_het_chstring_p+'\nA (sun)',
                    drawstyle='steps-mid')
    if Bepi:
        # ax.plot(sixs_p.index, sixs_p[sixs_ch_p], color='orange', linewidth=linewidth, label='BepiColombo/SIXS '+sixs_chstrings[sixs_ch_p]+f' side {sixs_side_p}', drawstyle='steps-mid')
        if len(sixs_df) > 0:
            ax.plot(sixs_df_p25.index, sixs_df_p25, color=sixs_color, linewidth=linewidth, label='BepiColombo\nSIXS '+sixs_p25_en_channel_string+f'\nside {sixs_side}', drawstyle='steps-mid')
    if Maven:
        ax.plot(pd.to_datetime(maven_p_new_1.index, format=maven_time_format), maven_p_new_1[f'E{maven_ch_p_new_1}'], color=maven_color, linewidth=linewidth, label=f'Maven {maven_p_channels_new_1[maven_ch_p_new_1]}', drawstyle='steps-mid')
    if SOLO:
        if het and (len(het_p) > 0):
            ax.plot(df_het_p.index, df_het_p, linewidth=linewidth, color=solo_het_color, label='SOLO\nHET '+het_chstring_p.replace('00 ', ' ')+f'\n{sector}', drawstyle='steps-mid')
    if STEREO:
        # if sept_p:
        #     if type(sept_ch_p) == list and len(sta_sept_avg_p) > 0:
        #         ax.plot(sta_sept_df_p.index, sta_sept_avg_p, color=stereo_sept_color, linewidth=linewidth, label='STEREO/SEPT '+sept_chstring_p+f' {sector}', drawstyle='steps-mid')
        #     elif type(sept_ch_p) == int:
        #         ax.plot(sta_sept_df_p.index, sta_sept_df_p[f'ch_{sept_ch_p}'], color=stereo_sept_color, linewidth=linewidth, label='STEREO/SEPT '+sta_sept_dict_p.loc[sept_ch_p]['ch_strings']+f' {sector}', drawstyle='steps-mid')
        if stereo_het:
            if len(sta_het_avg_p) > 0:
                ax.plot(sta_het_avg_p.index, sta_het_avg_p, color=stereo_het_color,
                        linewidth=linewidth, label='STEREO-A\nHET '+st_het_chstring_p, drawstyle='steps-mid')
        if let:
            str_ch = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}
            ax.plot(sta_let_df.index, sta_let_df[f'H_unsec_flux_{let_ch}'], color=stereo_let_color, linewidth=linewidth, label='STERE/LET '+let_chstring[let_ch], drawstyle='steps-mid')
    if SOHO:
        if erne:
            if type(erne_p_ch) == list and len(soho_erne_avg_p) > 0:
                ax.plot(soho_erne_avg_p.index, soho_erne_avg_p, color=soho_erne_color, linewidth=linewidth, label='SOHO\nERNE/HED '+soho_erne_chstring_p, drawstyle='steps-mid')
            elif type(erne_p_ch) == int:
                if len(soho_erne) > 0:
                    ax.plot(soho_erne.index, soho_erne[f'PH_{erne_p_ch}'], color=soho_erne_color, linewidth=linewidth, label='SOHO\nERNE/HED '+erne_chstring[erne_p_ch], drawstyle='steps-mid')
        # if ephin_p:
        #     ax.plot(ephin['date'], ephin[ephin_ch_p][0], color=soho_ephin_color, linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_p][1], drawstyle='steps-mid')
    # if WIND:
        # multiply by 1e6 to get per MeV
    #    ax.plot(wind3dp_p_df.index, wind3dp_p_df[f'FLUX_{wind3dp_ch_p}']*1e6, color=wind_color, linewidth=linewidth, label='Wind\n3DP '+str(round(wind3dp_p_df[f'ENERGY_{wind3dp_ch_p}'].mean()/1000., 2)) + ' keV', drawstyle='steps-mid')
    # ax.set_ylim(2.05e-5, 4.8e0)
    # ax.set_ylim(0.00033920545179055416, 249.08996960298424)
    ax.set_yscale('log')
    ax.set_ylabel(intensity_label)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='>25 MeV Protons/Ions')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  #, title='Protons/Ions')
    bbox = dict(boxstyle='round', fc='white', ec='black')
    ax.text(0.97, 0.90, 'Higher energy protons', bbox=bbox, transform=ax.transAxes, horizontalalignment='right')
    axnum = axnum+1

if plot_p_maven:
    if panels == 1:
        ax = axes
    else:
        ax = axes[axnum]
    if PSP:
        if len(psp_het) > 0:
            # ax.plot(psp_het.index, psp_het[f'A_H_Flux_{psp_het_ch_p}'], color=psp_het_color, linewidth=linewidth,
            #         label='PSP '+r"$\bf{(count\ rates)}$"+'\nISOIS-EPIHI-HET '+psp_het_energies['H_ENERGY_LABL'][psp_het_ch_p][0].replace(' ', '').replace('-', ' - ').replace('MeV', ' MeV')+'\nA (sun)',
            #         drawstyle='steps-mid')
            ax.plot(df_psp_het_p_maven.index, df_psp_het_p_maven, color=psp_het_color, linewidth=linewidth,
                    label='PSP '+'\nISOIS-EPIHI '+psp_het_chstring_p_maven+'\nA (sun)',
                    drawstyle='steps-mid')
    if Bepi:
        # ax.plot(sixs_p.index, sixs_p[sixs_ch_p], color='orange', linewidth=linewidth, label='BepiColombo/SIXS '+sixs_chstrings[sixs_ch_p]+f' side {sixs_side_p}', drawstyle='steps-mid')
        if len(sixs_df) > 0:
            ax.plot(sixs_df_p_maven.index, sixs_df_p_maven, color=sixs_color, linewidth=linewidth, label='BepiColombo\nSIXS '+sixs_p_maven_en_channel_string+f'\nside {sixs_side}', drawstyle='steps-mid')
    if Maven:
        ax.plot(pd.to_datetime(maven_p.index, format=maven_time_format), maven_p[f'E{maven_ch_p}'], color=maven_color, linewidth=linewidth, label=f'Maven {maven_p_channels[maven_ch_p]} keV', drawstyle='steps-mid')
        # ax.plot(pd.to_datetime(maven_p_new_2.index, format=maven_time_format), maven_p_new_2[f'E{maven_ch_p_new_2}'], color=maven_color, linewidth=linewidth, label=f'Maven {maven_p_channels_new_2[maven_ch_p_new_2]}', drawstyle='steps-mid')
    if SOLO:
        if het_maven:
            if het and (len(het_p) > 0):
                ax.plot(df_het_p_maven.index, df_het_p_maven, linewidth=linewidth, color=solo_het_color, label='SOLO\nHET '+het_chstring_p_maven.replace('00 ', ' ')+f'\n{sector}', drawstyle='steps-mid')
        if ept and (len(ept_p) > 0):
            ax.plot(df_ept_p.index, df_ept_p, linewidth=linewidth, color=solo_ept_color, label=f'SOLO\nEPT '+ept_chstring_p+f'\n{sector}', drawstyle='steps-mid')

            
    if STEREO:
        if sept_p:
            if type(sept_ch_p) == list and len(sta_sept_avg_p) > 0:
                ax.plot(sta_sept_df_p.index, sta_sept_avg_p, color=stereo_sept_color, linewidth=linewidth, label=f'STEREO\nSEPT {sept_chstring_p}\n{sector}', drawstyle='steps-mid')
            elif type(sept_ch_p) == int:
                ax.plot(sta_sept_df_p.index, sta_sept_df_p[f'ch_{sept_ch_p}'], color=stereo_sept_color, linewidth=linewidth, label=f'STEREO\nSEPT {sta_sept_dict_p.loc[sept_ch_p]['ch_strings']}\n{sector}', drawstyle='steps-mid')
        if stereo_het:
            if sta_het_maven:
                if len(sta_het_avg_p_maven) > 0:
                    ax.plot(sta_het_avg_p_maven.index, sta_het_avg_p_maven, color=stereo_het_color,
                            linewidth=linewidth, label='STEREO-A\nHET '+st_het_chstring_p_maven, drawstyle='steps-mid')
        if let:
            str_ch = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}
            ax.plot(sta_let_df.index, sta_let_df[f'H_unsec_flux_{let_ch}'], color=stereo_let_color, linewidth=linewidth, label='STERE/LET '+let_chstring[let_ch], drawstyle='steps-mid')
    if SOHO:
        if erne:
            if erne_maven:
                if type(erne_p_ch_maven) == int:
                    if len(soho_erne) > 0:
                        ax.plot(soho_erne.index, soho_erne[f'PH_{erne_p_ch_maven}'], color=soho_erne_color, linewidth=linewidth, label='SOHO\nERNE/HED '+erne_chstring[erne_p_ch_maven], drawstyle='steps-mid')
    # if SOHO:
    #     if erne:
    #         if type(erne_p_ch) == list and len(soho_erne_avg_p) > 0:
    #             ax.plot(soho_erne_avg_p.index, soho_erne_avg_p, color=soho_erne_color, linewidth=linewidth, label='SOHO\nERNE/HED '+soho_erne_chstring_p, drawstyle='steps-mid')
    #         elif type(erne_p_ch) == int:
    #             if len(soho_erne) > 0:
    #                 ax.plot(soho_erne.index, soho_erne[f'PH_{erne_p_ch}'], color=soho_erne_color, linewidth=linewidth, label='SOHO\nERNE/HED '+erne_chstring[erne_p_ch], drawstyle='steps-mid')
    #     # if ephin_p:
    #     #     ax.plot(ephin['date'], ephin[ephin_ch_p][0], color=soho_ephin_color, linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_p][1], drawstyle='steps-mid')
    # if WIND:
        # multiply by 1e6 to get per MeV
    #    ax.plot(wind3dp_p_df.index, wind3dp_p_df[f'FLUX_{wind3dp_ch_p}']*1e6, color=wind_color, linewidth=linewidth, label='Wind\n3DP '+str(round(wind3dp_p_df[f'ENERGY_{wind3dp_ch_p}'].mean()/1000., 2)) + ' keV', drawstyle='steps-mid')
    if WIND:
        if wind3dp_p:
            # multiply by 1e6 to get per MeV
            # ax.plot(wind3dp_p_df.index, wind3dp_p_df[f'FLUX_{wind3dp_ch_p}']*1e6, color=wind_color, linewidth=linewidth, label='Wind/3DP omni '+str(round(wind3dp_p_df[f'ENERGY_{wind3dp_ch_p}'].mean()/1000., 2)) + ' keV', drawstyle='steps-mid')
            ax.plot(wind3dp_p_df.index, wind3dp_p_df[f'FLUX_{wind3dp_ch_p}']*1e6, color=wind_color, linewidth=linewidth, label=f'Wind\n3DP {wind3dp_p_meta['channels_dict_df']['Bins_Text'].iloc[wind3dp_ch_p]}\nomni', drawstyle='steps-mid')    
    # ax.set_ylim(2.05e-5, 4.8e0)
    # ax.set_ylim(0.00033920545179055416, 249.08996960298424)
    ax.set_yscale('log')
    ax.set_ylabel(intensity_label)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='>25 MeV Protons/Ions')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.05))  # , title='Protons/Ions')
    bbox = dict(boxstyle='round', fc='white', ec='black')
    ax.text(0.97, 0.90, 'Lower energy protons/ions', bbox=bbox, transform=ax.transAxes, horizontalalignment='right')
    axnum = axnum+1


# pos = get_horizons_coord('Solar Orbiter', startdate, 'id')
# dist = np.round(pos.radius.value, 2)
# fig.suptitle(f'Solar Orbiter/EPD {sector} (R={dist} au)')
# fig.suptitle(f'Protons/Ions')
ax.set_xlim(startdate, enddate)
# ax.set_xlim(dt.datetime(2021, 10, 9, 6, 0), dt.datetime(2021, 10, 9, 11, 0))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel('Date / Time in year '+str(startdate.year))

for a in axes:
    a.axvline(dt.datetime(2023, 3, 13, 7, 14), lw=2, color=psp_het_color)
    a.axvline(dt.datetime(2023, 3, 13, 14, 52), lw=2, color=sixs_color)
    a.axvline(dt.datetime(2023, 3, 14, 1, 4), lw=2, color=solo_ept_color)
    a.axvline(dt.datetime(2023, 3, 15, 1, 16), lw=2, color=stereo_sept_color)
    a.axvline(dt.datetime(2023, 3, 15, 4, 1), lw=2, color='k')  # Wind

plt.tight_layout()
fig.subplots_adjust(hspace=0.1)
if save_fig:
    species = ''
    if plot_e_1 or plot_e_100:
        species = species+'e'
    if plot_p:
        species = species+'p'
    plt.savefig(outfile)
    plt.close()
    print('')
    print('Saved '+outfile)
else:
    plt.show()
