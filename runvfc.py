from vfcdevel.profilegen_v2 import VFC_and_spectra_v2
#from vfcdevel.profilegen_v2 import Profile_gen,Profile_gen_legacy
from vfcdevel.profilegen_v2 import Profile_gen

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
from varname import nameof

# custom libraries
import vfcdevel.forwardmodel as fm
from vfcdevel.utils import *
from vfcdevel.psd import mtm_psd
import vfcdevel.logsmooth as lsm

from scipy.interpolate import interp1d

from vfcdevel.make_pretty_figures import *

def core_details(core, df, namecore):

    if namecore=='ICORDA':
        label_core = 'ICORDA LDC'
        c_core = 'orchid'
        c_core_sm = 'purple'
        core_resolution = [[0, 1.939, 0.033], [1.939, 2.872, 0.040], [2.872, len(df), 0.0003]] # [z, z+dz, resol]
        grid = np.mean([0.033, 0.040])
        
    if namecore== 'subglacior':
        label_core = 'Subglacior LDC'
        c_core = 'limegreen' 
        c_core_sm = 'forestgreen'
        core_resolution = [[0, 1., 0.040], [1., 2., 0.038], [2., 3., 0.053], [3., len(df), 0.036]]
        grid = np.mean([0.040, 0.038, 0.053, 0.036])
        
    #core regular
    core_depth_regular = np.arange(core['Depth(m)'].min(), core['Depth(m)'].max(), grid)
    interp_d18O = interp1d(core['Depth(m)'], core['d18O'], kind='linear') 
    core_regular = pd.DataFrame({'Depth(m)': core_depth_regular,'d18O': interp_d18O(core_depth_regular)})
    
    return core_resolution, core_regular, grid, label_core, c_core, c_core_sm

def superplot(MODELDATA,OBSDATA, NAMECORE, NOISESCALE_MM=10, MIXINGSCALE_MM=40):
    
    # CORE
    core_resolution, core_regular, grid, label_core, c_core, c_core_sm = core_details(OBSDATA, MODELDATA, NAMECORE)
    
    freq_core,psd_core = mtm_psd(core_regular.d18O.dropna(),1/grid)
    psd_core_sm,freq_core_sm= lsm.logsmooth(psd_core,freq_core,0.05)[:2]    

    # RUN VFC: % mixing level % noise level
    
    c_VFC_no_diff = 'lavender' 
    c_VFC_diff_n10 = 'paleturquoise' 
    c_VFC_diff = 'skyblue'
    c_VFC_diff_sm = 'cadetblue'
    
    # Raw signal (0% noise, 0% mixing)
    VFC_nl0_ml0, spectra_nl0_ml0 =  VFC_and_spectra_v2(MODELDATA, core_resolution, mix_scale = MIXINGSCALE_MM, noise_scale = NOISESCALE_MM, nl=0, ml=0, regular_grid=grid)
    # # Noise level 0/20/40/60/80/100% (Mixing level 100%)
    VFC_nl0_ml100, spectra_nl0_ml100 =  VFC_and_spectra_v2(MODELDATA, core_resolution, mix_scale = MIXINGSCALE_MM, noise_scale = NOISESCALE_MM, nl=0, ml=1.0, regular_grid=grid)
    VFC_nl20_ml100, spectra_nl20_ml100 =  VFC_and_spectra_v2(MODELDATA, core_resolution, mix_scale = MIXINGSCALE_MM, noise_scale = NOISESCALE_MM, nl=0.2, ml=1.0, regular_grid=grid)
    VFC_nl40_ml100, spectra_nl40_ml100 =  VFC_and_spectra_v2(MODELDATA, core_resolution, mix_scale = MIXINGSCALE_MM, noise_scale = NOISESCALE_MM, nl=0.4, ml=1.0, regular_grid=grid)
    VFC_nl60_ml100, spectra_nl60_ml100 =  VFC_and_spectra_v2(MODELDATA, core_resolution, mix_scale = MIXINGSCALE_MM, noise_scale = NOISESCALE_MM, nl=0.6, ml=1.0, regular_grid=grid)
    VFC_nl80_ml100, spectra_nl80_ml100 =  VFC_and_spectra_v2(MODELDATA, core_resolution, mix_scale = MIXINGSCALE_MM, noise_scale = NOISESCALE_MM, nl=0.8, ml=1.0, regular_grid=grid)
    VFC_nl100_ml100, spectra_nl100_ml100 =  VFC_and_spectra_v2(MODELDATA, core_resolution, mix_scale = MIXINGSCALE_MM, noise_scale = NOISESCALE_MM, nl=1.0, ml=1.0, regular_grid=grid)

    
    # FIGURE VFC RAW 0% NOISE 0% MIXING
    
    plt.figure(figsize=(7, 14))
    
    VFC_nl_ml = VFC_nl0_ml0
    ax0 = plt.subplot(111)
    plt.step(OBSDATA['d18O'],-OBSDATA['Depth(m)'], where='mid', color= c_core , label=label_core, linewidth=4)
    plt.step(VFC_nl_ml['d18O no diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color = c_VFC_no_diff, label=None, linewidth=4)
    plt.step(VFC_nl_ml['d18O diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color = c_VFC_diff, label=None, linewidth=4)
    pretty_plot_vertical(ax0, 'Noise level = 0%, Mixing level = 0%', -3.1, 0, 'δ18O (‰)', 'Depth (m)')
    plt.xlim(-58, -44)
    
    
    plt.figure(figsize=(7, 7))
    
    spectra_nl_ml = spectra_nl0_ml0
    ax1 = plt.subplot(111)
    ax1.plot(freq_core,psd_core, color=c_core, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_core_sm, linestyle='--', linewidth=5)
    ax1.plot(spectra_nl_ml['freq_sm'],spectra_nl_ml['psd_sm'], color=c_VFC_no_diff, linewidth=5)
    for i in range(1, 11):
        ax1.plot(spectra_nl_ml['freq_diff{}'.format(i)],spectra_nl_ml['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
    ax1.plot(spectra_nl_ml['freq_diff_mean'],spectra_nl_ml['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
    ax1.plot(spectra_nl_ml['freq_diff_mean_sm'],spectra_nl_ml['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.title("Noise level = 0%, Mixing level = 0%", fontsize=15)

     
    # FIGURE NOISE EFFECT
    
    plt.figure(figsize=(25, 12))
    
    VFC_nl_ml = VFC_nl0_ml100
    ax0 = plt.subplot(161)
    plt.step(OBSDATA['d18O'],-OBSDATA['Depth(m)'], where = 'mid', color=c_core, label=label_core, linewidth=4)
    plt.step(VFC_nl_ml['d18O no diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_no_diff, label=None, linewidth=4)
    plt.step(VFC_nl_ml['d18O diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_diff, label=None, linewidth=4)
    pretty_plot_vertical(ax0, 'Noise 0%', -3.1, 0, 'δ18O (‰)', 'Depth (m)')
    plt.xlim(-58, -44)
    
    VFC_nl_ml = VFC_nl20_ml100
    ax0 = plt.subplot(162, sharex=ax0)
    plt.step(OBSDATA['d18O'],-OBSDATA['Depth(m)'], where = 'mid', color=c_core, label=label_core, linewidth=4)
    plt.step(VFC_nl_ml['d18O no diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_no_diff, label=None, linewidth=4)
    plt.step(VFC_nl_ml['d18O diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_diff, label=None, linewidth=4)
    pretty_plot_vertical(ax0, 'Noise 20%', -3.1, 0, 'δ18O (‰)', '')
    
    VFC_nl_ml = VFC_nl40_ml100
    ax0 = plt.subplot(163, sharex=ax0)
    plt.step(OBSDATA['d18O'],-OBSDATA['Depth(m)'], where = 'mid', color=c_core, label=label_core, linewidth=4)
    plt.step(VFC_nl_ml['d18O no diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_no_diff, label=None, linewidth=4)
    plt.step(VFC_nl_ml['d18O diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_diff, label=None, linewidth=4)
    pretty_plot_vertical(ax0, 'Noise 40%', -3.1, 0, 'δ18O (‰)', '')
    
    VFC_nl_ml = VFC_nl60_ml100
    ax0 = plt.subplot(164, sharex=ax0)
    plt.step(OBSDATA['d18O'],-OBSDATA['Depth(m)'], where = 'mid', color=c_core, label=label_core, linewidth=4)
    plt.step(VFC_nl_ml['d18O no diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_no_diff, label=None, linewidth=4)
    plt.step(VFC_nl_ml['d18O diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_diff, label=None, linewidth=4)
    pretty_plot_vertical(ax0, 'Noise 60%', -3.1, 0, 'δ18O (‰)', '')
    
    VFC_nl_ml = VFC_nl80_ml100
    ax0 = plt.subplot(165, sharex=ax0)
    plt.step(OBSDATA['d18O'],-OBSDATA['Depth(m)'], where = 'mid', color=c_core, label=label_core, linewidth=4)
    plt.step(VFC_nl_ml['d18O no diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_no_diff, label=None, linewidth=4)
    plt.step(VFC_nl_ml['d18O diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_diff, label=None, linewidth=4)
    pretty_plot_vertical(ax0, 'Noise 80%', -3.1, 0, 'δ18O (‰)', '')
    
    VFC_nl_ml = VFC_nl100_ml100
    ax0 = plt.subplot(166, sharex=ax0)
    plt.step(OBSDATA['d18O'],-OBSDATA['Depth(m)'], where = 'mid', color=c_core, label=label_core, linewidth=4)
    plt.step(VFC_nl_ml['d18O no diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_no_diff, label=None, linewidth=4)
    plt.step(VFC_nl_ml['d18O diff'],-VFC_nl_ml['Depth(m)'], where = 'mid', color=c_VFC_diff, label=None, linewidth=4)
    pretty_plot_vertical(ax0, 'Noise 100%', -3.1, 0, 'δ18O (‰)', '')
    
    plt.figure(figsize=(25,4))
    
    spectra_nl_ml = spectra_nl0_ml100
    ax1 = plt.subplot(161)
    ax1.plot(freq_core,psd_core, color=c_core, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_core_sm, linestyle='--', linewidth=5)
    ax1.plot(spectra_nl_ml['freq_sm'],spectra_nl_ml['psd_sm'], color=c_VFC_no_diff, linewidth=5)
    for i in range(1, 11):
        ax1.plot(spectra_nl_ml['freq_diff{}'.format(i)],spectra_nl_ml['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
    ax1.plot(spectra_nl_ml['freq_diff_mean'],spectra_nl_ml['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
    ax1.plot(spectra_nl_ml['freq_diff_mean_sm'],spectra_nl_ml['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.ylim(0.00007, 4)
    plt.title("Noise level 0%", fontsize=15)
    
    spectra_nl_ml = spectra_nl20_ml100
    ax1 = plt.subplot(162, sharex=ax1, sharey=ax1)
    ax1.plot(freq_core,psd_core, color=c_core, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_core_sm, linestyle='--', linewidth=5)
    ax1.plot(spectra_nl_ml['freq_sm'],spectra_nl_ml['psd_sm'], color=c_VFC_no_diff, linewidth=5)
    for i in range(1, 11):
        ax1.plot(spectra_nl_ml['freq_diff{}'.format(i)],spectra_nl_ml['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
    ax1.plot(spectra_nl_ml['freq_diff_mean'],spectra_nl_ml['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
    ax1.plot(spectra_nl_ml['freq_diff_mean_sm'],spectra_nl_ml['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.title("Noise level 20%", fontsize=15)
    
    spectra_nl_ml = spectra_nl40_ml100
    ax1 = plt.subplot(163, sharex=ax1, sharey=ax1)
    ax1.plot(freq_core,psd_core, color=c_core, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_core_sm, linestyle='--', linewidth=5)
    ax1.plot(spectra_nl_ml['freq_sm'],spectra_nl_ml['psd_sm'], color=c_VFC_no_diff, linewidth=5)
    for i in range(1, 11):
        ax1.plot(spectra_nl_ml['freq_diff{}'.format(i)],spectra_nl_ml['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
    ax1.plot(spectra_nl_ml['freq_diff_mean'],spectra_nl_ml['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
    ax1.plot(spectra_nl_ml['freq_diff_mean_sm'],spectra_nl_ml['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.title("Noise level 40%", fontsize=15)
    
    spectra_nl_ml = spectra_nl60_ml100
    ax1 = plt.subplot(164, sharex=ax1, sharey=ax1)
    ax1.plot(freq_core,psd_core, color=c_core, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_core_sm, linestyle='--', linewidth=5)
    ax1.plot(spectra_nl_ml['freq_sm'],spectra_nl_ml['psd_sm'], color=c_VFC_no_diff, linewidth=5)
    for i in range(1, 11):
        ax1.plot(spectra_nl_ml['freq_diff{}'.format(i)],spectra_nl_ml['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
    ax1.plot(spectra_nl_ml['freq_diff_mean'],spectra_nl_ml['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
    ax1.plot(spectra_nl_ml['freq_diff_mean_sm'],spectra_nl_ml['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.title("Noise level 60%", fontsize=15)
    
    spectra_nl_ml = spectra_nl80_ml100
    ax1 = plt.subplot(165, sharex=ax1, sharey=ax1)
    ax1.plot(freq_core,psd_core, color=c_core, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_core_sm, linestyle='--', linewidth=5)
    ax1.plot(spectra_nl_ml['freq_sm'],spectra_nl_ml['psd_sm'], color=c_VFC_no_diff, linewidth=5)
    for i in range(1, 11):
        ax1.plot(spectra_nl_ml['freq_diff{}'.format(i)],spectra_nl_ml['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
    ax1.plot(spectra_nl_ml['freq_diff_mean'],spectra_nl_ml['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
    ax1.plot(spectra_nl_ml['freq_diff_mean_sm'],spectra_nl_ml['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.title("Noise level 80%", fontsize=15)
    
    spectra_nl_ml = spectra_nl100_ml100
    ax1 = plt.subplot(166, sharex=ax1, sharey=ax1)
    ax1.plot(freq_core,psd_core, color=c_core, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_core_sm, linestyle='--', linewidth=5)
    ax1.plot(spectra_nl_ml['freq_sm'],spectra_nl_ml['psd_sm'], color=c_VFC_no_diff, linewidth=5)
    for i in range(1, 11):
        ax1.plot(spectra_nl_ml['freq_diff{}'.format(i)],spectra_nl_ml['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
    ax1.plot(spectra_nl_ml['freq_diff_mean'],spectra_nl_ml['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
    ax1.plot(spectra_nl_ml['freq_diff_mean_sm'],spectra_nl_ml['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.title("Noise level 100%", fontsize=15)
