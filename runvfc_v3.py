from vfcdevel.profilegen_v3 import VFC_and_spectra
#from vfcdevel.profilegen_v2 import Profile_gen,Profile_gen_legacy
from vfcdevel.profilegen_v3 import Profile_gen


from vfcdevel.pretty_plot import *

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

# generic libraries
from pathlib import Path
import netCDF4
import os
import numpy as np
import datetime as datetime
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from inspect import getmembers, isfunction
import pickle as pkl
import scipy
import copy
import matplotlib as mpl
#from varname import nameof

# custom libraries
import vfcdevel.forwardmodel as fm
from vfcdevel.utils import *
from vfcdevel.spectralanalysis import mtm_psd
import vfcdevel.logsmooth as lsm

from scipy.interpolate import interp1d


def core_details(core, df, corename):

    if corename=='ICORDA':
        label_core = 'ICORDA LDC'
        c_core = 'orchid' # core color
        c_core_sm = 'purple' # core color smoothed
        core_resolution = [[0, 1.939, 0.033], [1.939, 2.872, 0.040], [2.872, np.max(core['Depth(m)']), 0.0003]] # [z, z+dz, resol]
        grid = np.mean([0.033, 0.040])
        
    if corename== 'subglacior':
        label_core = 'Subglacior LDC'
        c_core = 'limegreen' 
        c_core_sm = 'forestgreen'
        core_resolution = [[0, 1., 0.040], [1., 2., 0.038], [2., 3., 0.053], [3., np.max(core['Depth(m)']), 0.036]]
        grid = np.mean([0.040, 0.038, 0.053, 0.036])
        
    #core regular
    core_depth_regular = np.arange(core['Depth(m)'].min(), core['Depth(m)'].max(), grid)
    interp_d18O = interp1d(core['Depth(m)'], core['d18O'], kind='linear') 
    core_regular = pd.DataFrame({'Depth(m)': core_depth_regular,'d18O': interp_d18O(core_depth_regular)})
    
    return core_resolution, core_regular, grid, label_core, c_core, c_core_sm

def superplot(MODELDATA,OBSDATA, CORENAME, noise_scale_m=10*1e-3, mixing_scale_m=40*1e-3,vfcres = 1e-3):
    
    # CORE
    core_resolution, core_regular, grid, label_core, c_core, c_core_sm = core_details(OBSDATA, MODELDATA, CORENAME)
    
    freq_core,psd_core = mtm_psd(core_regular.d18O.dropna(),1/grid)
    psd_core_sm,freq_core_sm= lsm.logsmooth(psd_core,freq_core,0.05)[:2]    

    # RUN VFC: % mixing level % noise level
    
    c_VFC_no_diff = 'lavender' 
    c_VFC_diff_n10 = 'paleturquoise' 
    c_VFC_diff = 'skyblue'
    c_VFC_diff_sm = 'cadetblue'

    # dictionaries that will contain all the dataframes
    VFC_dic = {}
    spectra_dic = {}

    for pair in [(0,0),(0,1),(.20,1),(.40,1),(.60,1),(.80,1),(1,1)]:
        print(pair)
        nl,ml = pair
        #VFC, spectra =  VFC_and_spectra(MODELDATA, core_resolution, mix_scale = MIXINGSCALE_M, noise_scale = NOISESCALE_M, nl=nl, ml=ml,fftres=1e-2,vfcres=vfcres)
        VFC, spectra =  VFC_and_spectra(MODELDATA, core_resolution, 
                                        mixing_scale_m = mixing_scale_m, noise_scale_m = noise_scale_m, noise_level=nl, mixing_level=ml,fftres=grid,vfcres=vfcres)
        VFC_dic[pair] = VFC
        spectra_dic[pair] = spectra
        
    plt.figure(figsize=(7, 14))
    
    VFC = VFC_dic[(0,0)]
    ax0 = plt.subplot(111)
    plt.step(OBSDATA['d18O'],-OBSDATA['Depth(m)'], where='mid', color= c_core , label=label_core, linewidth=4)
    plt.step(VFC['d18O'],-VFC.index, where = 'mid', color = c_VFC_no_diff, label=None, linewidth=4)
    plt.step(VFC['d18O_diff'],-VFC.index, where = 'mid', color = c_VFC_diff, label=None, linewidth=4)
    pretty_plot_vertical(ax0, 'Noise level = 0%, Mixing level = 0%', -3.1, 0, 'δ18O (‰)', 'Depth (m)')
    plt.xlim(-58, -44)
    
    
    plt.figure(figsize=(7, 7))
    
    spectra = spectra_dic[(0,0)]
    ax1 = plt.subplot(111)
    ax1.plot(freq_core,psd_core, color=c_core, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_core_sm, linestyle='--', linewidth=5)
    ax1.plot(spectra['freq_sm'],spectra['psd_sm'], color=c_VFC_no_diff, linewidth=5)
    for i in range(10):
        if 'freq_diff{}'.format(i) in spectra.columns:
            ax1.plot(spectra['freq_diff{}'.format(i)],spectra['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
    ax1.plot(spectra['freq_diff_mean'],spectra['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
    ax1.plot(spectra['freq_diff_mean_sm'],spectra['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.title("Noise level = 0%, Mixing level = 0%", fontsize=15)

    #return

     
    # FIGURE NOISE EFFECT
    
    plt.figure(figsize=(25, 12))

    for i,pair in enumerate([(0,1),(.20,1),(.40,1),(.60,1),(.80,1),(1,1)]):
        nl,ml = pair
        VFC_nl_ml = VFC_dic[pair]
    
        ax0 = plt.subplot(161+i)
        plt.step(OBSDATA['d18O'],-OBSDATA['Depth(m)'], where = 'mid', color=c_core, label=label_core, linewidth=4)
        plt.step(VFC_nl_ml['d18O'],-VFC_nl_ml.index, where = 'mid', color=c_VFC_no_diff, label=None, linewidth=4)
        plt.step(VFC_nl_ml['d18O_diff'],-VFC_nl_ml.index, where = 'mid', color=c_VFC_diff, label=None, linewidth=4)
        pretty_plot_vertical(ax0, 'Noise {0}%'.format(int(100*nl)), -3.1, 0, 'δ18O (‰)', '')
        plt.xlim(-58, -44)
    
    plt.figure(figsize=(25,4))

    for i,pair in enumerate([(0,1),(.20,1),(.40,1),(.60,1),(.80,1),(1,1)]):
        nl,ml = pair
    
        spectra_nl_ml = spectra_dic[pair]
        
        ax1 = plt.subplot(161+i)
        ax1.plot(freq_core,psd_core, color=c_core, label=label_core, linewidth = 2)
        ax1.plot(freq_core_sm,psd_core_sm, color=c_core_sm, linestyle='--', linewidth=5)
        ax1.plot(spectra_nl_ml['freq_sm'],spectra_nl_ml['psd_sm'], color=c_VFC_no_diff, linewidth=5)
        for i in range(10):
            if 'freq_diff{}'.format(i) in spectra.columns:
                ax1.plot(spectra_nl_ml['freq_diff{}'.format(i)],spectra_nl_ml['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
        ax1.plot(spectra_nl_ml['freq_diff_mean'],spectra_nl_ml['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
        ax1.plot(spectra_nl_ml['freq_diff_mean_sm'],spectra_nl_ml['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        plt.ylim(0.00007, 4)
        plt.title("Noise level {0}%".format(int(100*nl)), fontsize=15)