from vfcdevel.profilegen_v2 import VFC_and_spectra_v2
from vfcdevel.profilegen_v2 import Profile_gen,Profile_gen_legacy

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

# custom libraries
import vfcdevel.forwardmodel as fm
from vfcdevel.utils import *
from vfcdevel.psd import mtm_psd
import vfcdevel.logsmooth as lsm

from scipy.interpolate import interp1d

from vfcdevel.plot import *



def superplot(MODELDATA,OBSDATA):

    # please change all ICORDA instance with OBSDATA and df with MODELDATA
    
    ICORDA = OBSDATA

    df = MODELDATA

    #%% Colors for figures
    
    c_ICORDA = 'orchid'
    c_ICORDA_sm = 'purple'
    c_subglacior = 'limegreen'
    c_subglacior_sm = 'forestgreen'
    c_VFC_no_diff = 'lavender' 
    c_VFC_diff_n10 = 'paleturquoise' 
    c_VFC_diff = 'skyblue'
    c_VFC_diff_sm = 'cadetblue'
        
    #%% PARAMETERS
    
    which_core = 'ICORDA'
    
    grid = 0.035
    n_scale_mm = 20 #10 #20
    m_scale_mm = 160 #40 #80 #160
    
    #%% RUN : % mixing level % noise level
    
    # Raw signal (0% noise, 0% mixing)
    VFC_nl0_ml0, spectra_nl0_ml0 =  VFC_and_spectra_v2(df, mix_scale = m_scale_mm, noise_scale = n_scale_mm, nl=0, ml=0, regular_grid=grid)
    # Noise level 0/20/40/60/80/100% (Mixing level 100%)
    # VFC_nl0_ml100, spectra_nl0_ml100 =  VFC_and_spectra_v2(df, mix_scale = m_scale_mm, noise_scale = n_scale_mm, nl=0, ml=1.0, regular_grid=grid)
    # VFC_nl20_ml100, spectra_nl20_ml100 =  VFC_and_spectra_v2(df, mix_scale = m_scale_mm, noise_scale = n_scale_mm, nl=0.2, ml=1.0, regular_grid=grid)
    # VFC_nl40_ml100, spectra_nl40_ml100 =  VFC_and_spectra_v2(df, mix_scale = m_scale_mm, noise_scale = n_scale_mm, nl=0.4, ml=1.0, regular_grid=grid)
    # VFC_nl60_ml100, spectra_nl60_ml100 =  VFC_and_spectra_v2(df, mix_scale = m_scale_mm, noise_scale = n_scale_mm, nl=0.6, ml=1.0, regular_grid=grid)
    # VFC_nl80_ml100, spectra_nl80_ml100 =  VFC_and_spectra_v2(df, mix_scale = m_scale_mm, noise_scale = n_scale_mm, nl=0.8, ml=1.0, regular_grid=grid)
    # VFC_nl100_ml100, spectra_nl100_ml100 =  VFC_and_spectra_v2(df, mix_scale = m_scale_mm, noise_scale = n_scale_mm, nl=1.0, ml=1.0, regular_grid=grid)
    
    
    ############ ICORDA regular
    ICORDA_depth_regular = np.arange(ICORDA['Depth(m)'].min(), ICORDA['Depth(m)'].max(), grid)
    interp_age = interp1d(ICORDA['Depth(m)'], ICORDA['Age_AD'], kind='linear')  ; interp_d18O = interp1d(ICORDA['Depth(m)'], ICORDA['d18O'], kind='linear') 
    ICORDA_regular = pd.DataFrame({'Depth(m)': ICORDA_depth_regular, 'Age_AD': interp_age(ICORDA_depth_regular) ,'d18O': interp_d18O(ICORDA_depth_regular)})
    
    freq_ICORDA,psd_ICORDA = mtm_psd(ICORDA_regular.d18O.dropna(),1/grid)
    psd_ICORDA_sm,freq_ICORDA_sm= lsm.logsmooth(psd_ICORDA,freq_ICORDA,0.05)[:2]
    
    ############ Subglacior regular
    #subglacior_depth_regular = np.arange(subglacior['Depth(m)'].min(), subglacior['Depth(m)'].max(), grid)
    #interp_age = interp1d(subglacior['Depth(m)'], subglacior['Age_timescale(ka BP)'], kind='linear')  ; interp_d18O = interp1d(subglacior['Depth(m)'], subglacior['d18O'], kind='linear') 
    #subglacior_regular = pd.DataFrame({'Depth(m)': subglacior_depth_regular, 'Age_AD': interp_age(subglacior_depth_regular) ,'d18O': interp_d18O(subglacior_depth_regular)})
    
    #freq_subglacior,psd_subglacior = mtm_psd(subglacior_regular.d18O.dropna(),1/grid)
    #psd_subglacior_sm,freq_subglacior_sm= lsm.logsmooth(psd_subglacior,freq_subglacior,0.05)[:2]
    
    if which_core=='ICORDA':
        core = ICORDA           ;  label_core = 'ICORDA LDC'
        freq_core = freq_ICORDA ;  freq_core_sm = freq_ICORDA_sm
        psd_core = psd_ICORDA   ;  psd_core_sm = psd_ICORDA_sm
        c = c_ICORDA            ;  c_sm = c_ICORDA_sm
    if which_core == 'subglacior':
        core = subglacior           ;  label_core = 'Subglacior LDC'
        freq_core = freq_subglacior ;  freq_core_sm = freq_subglacior_sm
        psd_core = psd_subglacior   ;  psd_core_sm = psd_subglacior_sm
        c = c_subglacior            ;  c_sm = c_subglacior_sm
    
    
    #%% FIGURE VFC RAW 0% NOISE 0% MIXING
    
    plt.figure(figsize=(7, 14))
    
    VFC_nl_ml = VFC_nl0_ml0
    ax0 = plt.subplot(111)
    
    plot_stairsteps(core['d18O'],-core['Depth(m)'], c_core= c , label=label_core, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O no diff']),-np.array(VFC_nl_ml['Depth(m)']),c_VFC_no_diff, label=None, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O diff']),-np.array(VFC_nl_ml['Depth(m)']), c_VFC_diff, label=None, line_width=4)
    pretty_plot_vertical(ax0, 'Noise level = 0%, Mixing level = 0%', -3.1, 0, 'δ18O (‰)', 'Depth (m)')
    plt.xlim(-58, -44)
    
    
    plt.figure(figsize=(7, 7))
    
    spectra_nl_ml = spectra_nl0_ml0
    ax1 = plt.subplot(111)
    ax1.plot(freq_core,psd_core, color=c, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_sm, linestyle='--', linewidth=5)
    ax1.plot(spectra_nl_ml['freq_sm'],spectra_nl_ml['psd_sm'], color=c_VFC_no_diff, linewidth=5)
    for i in range(1, 11):
        ax1.plot(spectra_nl_ml['freq_diff{}'.format(i)],spectra_nl_ml['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
    ax1.plot(spectra_nl_ml['freq_diff_mean'],spectra_nl_ml['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
    ax1.plot(spectra_nl_ml['freq_diff_mean_sm'],spectra_nl_ml['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.title("Noise level = 0%, Mixing level = 0%", fontsize=15)
    
    STOP
    #%% FIGURE NOISE EFFECT
    
    
    plt.figure(figsize=(25, 12))
    
    VFC_nl_ml = VFC_nl0_ml100
    ax0 = plt.subplot(161)
    plot_stairsteps(core['d18O'],-core['Depth(m)'], c_core= c , label=label_core, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O no diff']),-np.array(VFC_nl_ml['Depth(m)']),c_VFC_no_diff, label=None, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O diff']),-np.array(VFC_nl_ml['Depth(m)']), c_VFC_diff, label=None, line_width=4)
    pretty_plot_vertical(ax0, 'Noise 0%', -3.1, 0, 'δ18O (‰)', 'Depth (m)')
    plt.xlim(-58, -44)
    
    VFC_nl_ml = VFC_nl20_ml100
    ax0 = plt.subplot(162, sharex=ax0)
    plot_stairsteps(core['d18O'],-core['Depth(m)'], c_core= c , label=label_core, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O no diff']),-np.array(VFC_nl_ml['Depth(m)']),c_VFC_no_diff, label=None, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O diff']),-np.array(VFC_nl_ml['Depth(m)']), c_VFC_diff, label=None, line_width=4)
    pretty_plot_vertical(ax0, 'Noise 20%', -3.1, 0, 'δ18O (‰)', '')
    
    VFC_nl_ml = VFC_nl40_ml100
    ax0 = plt.subplot(163, sharex=ax0)
    plot_stairsteps(core['d18O'],-core['Depth(m)'], c_core= c , label=label_core, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O no diff']),-np.array(VFC_nl_ml['Depth(m)']),c_VFC_no_diff, label=None, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O diff']),-np.array(VFC_nl_ml['Depth(m)']), c_VFC_diff, label=None, line_width=4)
    pretty_plot_vertical(ax0, 'Noise 40%', -3.1, 0, 'δ18O (‰)', '')
    
    VFC_nl_ml = VFC_nl60_ml100
    ax0 = plt.subplot(164, sharex=ax0)
    plot_stairsteps(core['d18O'],-core['Depth(m)'], c_core= c , label=label_core, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O no diff']),-np.array(VFC_nl_ml['Depth(m)']),c_VFC_no_diff, label=None, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O diff']),-np.array(VFC_nl_ml['Depth(m)']), c_VFC_diff, label=None, line_width=4)
    pretty_plot_vertical(ax0, 'Noise 60%', -3.1, 0, 'δ18O (‰)', '')
    
    VFC_nl_ml = VFC_nl80_ml100
    ax0 = plt.subplot(165, sharex=ax0)
    plot_stairsteps(core['d18O'],-core['Depth(m)'], c_core= c , label=label_core, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O no diff']),-np.array(VFC_nl_ml['Depth(m)']),c_VFC_no_diff, label=None, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O diff']),-np.array(VFC_nl_ml['Depth(m)']), c_VFC_diff, label=None, line_width=4)
    pretty_plot_vertical(ax0, 'Noise 80%', -3.1, 0, 'δ18O (‰)', '')
    
    VFC_nl_ml = VFC_nl100_ml100
    ax0 = plt.subplot(166, sharex=ax0)
    plot_stairsteps(core['d18O'],-core['Depth(m)'], c_core= c , label=label_core, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O no diff']),-np.array(VFC_nl_ml['Depth(m)']),c_VFC_no_diff, label=None, line_width=4)
    plot_stairsteps(np.array(VFC_nl_ml['d18O diff']),-np.array(VFC_nl_ml['Depth(m)']), c_VFC_diff, label=None, line_width=4)
    pretty_plot_vertical(ax0, 'Noise 100%', -3.1, 0, 'δ18O (‰)', '')
    
    plt.figure(figsize=(25,4))
    
    spectra_nl_ml = spectra_nl0_ml100
    ax1 = plt.subplot(161)
    ax1.plot(freq_core,psd_core, color=c, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_sm, linestyle='--', linewidth=5)
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
    ax1.plot(freq_core,psd_core, color=c, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_sm, linestyle='--', linewidth=5)
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
    ax1.plot(freq_core,psd_core, color=c, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_sm, linestyle='--', linewidth=5)
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
    ax1.plot(freq_core,psd_core, color=c, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_sm, linestyle='--', linewidth=5)
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
    ax1.plot(freq_core,psd_core, color=c, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_sm, linestyle='--', linewidth=5)
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
    ax1.plot(freq_core,psd_core, color=c, label=label_core, linewidth = 2)
    ax1.plot(freq_core_sm,psd_core_sm, color=c_sm, linestyle='--', linewidth=5)
    ax1.plot(spectra_nl_ml['freq_sm'],spectra_nl_ml['psd_sm'], color=c_VFC_no_diff, linewidth=5)
    for i in range(1, 11):
        ax1.plot(spectra_nl_ml['freq_diff{}'.format(i)],spectra_nl_ml['psd_diff{}'.format(i)], color=c_VFC_diff_n10, alpha=0.3)
    ax1.plot(spectra_nl_ml['freq_diff_mean'],spectra_nl_ml['psd_diff_mean'], color=c_VFC_diff, linewidth=3, label = 'VFC diff (n=10)')
    ax1.plot(spectra_nl_ml['freq_diff_mean_sm'],spectra_nl_ml['psd_diff_mean_sm'], color=c_VFC_diff_sm, linestyle='--', linewidth=5, label = 'VFC diff (n=10)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.title("Noise level 100%", fontsize=15)
