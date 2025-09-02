#%% Functions VFC

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
from vfcdevel.spectralanalysis import mtm_psd
import vfcdevel.logsmooth as lsm

from vfcdevel.isotopes import temp_to_iso_DC

from vfcdevel.utils import block_average_OLD3 as block_average

from scipy.interpolate import interp1d



def Profile_gen(Date,Temp,Prec,Precip_d18O, c_resolution, storage_diffusion_cm, rho, mixing_level=0.1, noise_level=0.1, mixing_scale_mm = 40, noise_scale_mm = 10):
    
    
    # Generates the profile of isotopic composition of a snow column
    #
    # This calculates the isotopic composition of polar firn for the stable
    # water isotopes oxygen-18, depending on site-specific parameters, as well
    # as the temperature and precipitation time series. 
    #
    # Date : Datetime vector object, describing the temperature (Temp) and
    #        precipitation (Prec) time series
    # Temp : Numeric vector of temperature (°C), same length as Date
    # Prec : Numeric vector of precipitation amount (kg.m-2 or mm w.e.), 
    #        same length as Date
    # rho  : Local firn density (kg.m-3), single value
    #
    # This routine use the temperature and precipitation time series to create
    # a virtual ice cores that compiles precipitation intermittency and
    # diffusion. 
    # First, the temperature time serie is converted into an isotopic
    # composition of precipitation using common coefficient (alpha and beta,
    # Stenni et al, 2016). For a specific site where these parameters are
    # available, these can be adjusted. 
    # Then, the intermittent virtual core is computed by adding layers of
    # snow with a width determined by the precipitation amount associated with
    # each events, and an isotopic composition calculated from the temperature
    # time serie. This intermittent virtual core is interpolated onto a regular
    # grid, using a constant step of 1 mm, which can lead to very large
    # variables for important accumulation sites or very long time series. The
    # resolution at which the core is interpolated should remain small compared
    # to the diffusion length (see below). 
    # Then, the isotopic profile is diffused, and the diffused virtual core is
    # generated. Finally, the intermittent (d18O_int) and the diffused 
    # (d18O_diff) virtual core profiles are blocked averaged to a resolution 
    # of 1 cm, a value similar to the thickness of samples from ice core 
    # records.
    #
    # This function was created in the framework of the manuscript Casado et
    # al, Climatic information archived in ice cores: impact of intermittency
    # and diffusion on the recorded isotopic signal in Antarctica, Climate of
    # the Past (2020)
    
    # a wrapper that does the in principle the same as the original matlab version
    # profile gen, diffuse and block average

    #Preamble;
    #To avoid numerical errors leading to doublons of datapoints with the same depth due to numerical approximation
    # (legacy (superfluous?), for now to have the same result as the matlab version)
    
    keep = (Prec>1e-10)
    Date = Date[keep]
    Temp = Temp.loc[keep]
    Prec = Prec.loc[(keep)]
    if not Precip_d18O is None: Precip_d18O = Precip_d18O.loc[keep]
    
    
    
    #Generates the profile of isotopic composition of a snow column
    # <!> generates profile at the mm resolution, no block average (see profile_gen_legacy)
    ddays = np.diff(Date).astype('timedelta64[D]').astype(int) # convert to number of days, and we will assume precip is in units of per days
    ddays = np.concatenate([[ddays[0]],ddays]) # restore first value, assume it is similar to second value (daily, monthly...)

    ## Data preparation
    accu = np.sum(Prec*ddays)/((Date[-1]-Date[0]).days/(365)) # Calculation of the accumulation rate in mm.a-1 w.e.)
    Prec = Prec/rho*997/1000 #Conversion to meters and snow equivalent
    Prec.loc[(Prec<=0)] = 0 #Negative precipitation cannot be taken into account. In the case of true accumulation, where negative values could occur, changes to the script need to be made.
    print('accu =', accu)

    ## Compute precipitation intermittent depth serie    
    date_raw = Date.to_numpy()
    if Precip_d18O is None:
        # if d18O is not provided; compute from temperature
        #d18O_raw = temp_to_d18O_DC(Temp).to_numpy()
        temp_to_iso_DC(Temp,'18O',fit='Mathieu').to_numpy()
    else:
        #else, just copy the input d18O
        d18O_raw = Precip_d18O.to_numpy()
        
    depth_raw = np.cumsum(Prec.to_numpy()[::-1]*ddays)[::-1]  # to meters
    #depth_raw_mean = depth_raw[:-1] + np.diff(depth_raw)/2; # middle grid points # I don't use this at the moment
    

    ## Sublimation (removing isotopes, no snow layer thickness change, only density)
    ## Condensation from the atmosphere
    ## Frost at the surface
    ## Condensation from the snow below

    ################################################################## Interpolation to a regular grid ###############################################################
    step = 0.001; # grid of 1mm
    Depth_tot = depth_raw[0]
    
    depth_even = np.arange(step,Depth_tot-step,step);

    d18O_ini = np.interp(depth_even,depth_raw[::-1],d18O_raw[::-1]); # invert depth_raw to be increasing
    date_even = float_time_interp(depth_even,depth_raw[::-1],Date[::-1]) # date even is now decreasing
    
    df = pd.DataFrame({'depth_even': depth_even, 'd18O_ini': d18O_ini}) 


    #### test with block average
    dfi = pd.DataFrame({'d18O_ini':d18O_raw},index=depth_raw)
    df = block_average_OLD(dfi,step)
    depth_even = df.index.to_numpy()
    df['depth_even'] = depth_even

    d18O_ini = df['d18O_ini']
    date_even = float_time_interp(depth_even,depth_raw[::-1],Date[::-1])
    
    ####################################################################### Generate white noise #####################################################################
    sig_d18O = np.nanstd(d18O_ini); #d18O_int
    m_d18O = np.nanmean(d18O_ini);
    wn18O = np.full(d18O_ini.shape,np.nan)
    
    noise_scale = noise_scale_mm
    # RANDOM AVEC DISTRIBUTION GAUSSIENNE
    wn18O = np.zeros_like(d18O_ini, dtype=float)
    for i in range(len(d18O_ini) // noise_scale):
        wn18O[noise_scale * i : noise_scale * (i + 1)] = np.random.randn()
    ind = np.argwhere(np.isnan(wn18O))
    wn18O[ind]= np.random.randn()
    
    ################################################################# VFC[d18O ini & white noise] (Laepple) ###########################################################

    # I comment this factor for now
    #factor = 1/np.sqrt(noise_scale/10)
    factor = 1
    d18O_noise_Laepple = np.sqrt(1. - noise_level)*d18O_ini + np.sqrt(noise_level)*factor*sig_d18O*wn18O

    
    ######################################################################### Mixing ##################################################################################
    # mixing is implemented as a rolling window with width given by the mixing_scale in mm snow
    # rolling window is centered (value can be influenced by the snow coming before and after)
    # min_periods = 1 so the more recent snow will only be mixed with the snow underneath it (otherwise we would get nan values at the top which is wrong)
    # (technically we should write min_periods = int(mixing_scale_mm/res/2))
    
    df = pd.DataFrame({'depth_even': depth_even, 'd18O_noise': d18O_noise_Laepple})
    df['d18O_rolling_avg'] = df['d18O_noise'].rolling(window=mixing_scale_mm, center=True,min_periods=1).mean()  # d18O-rolling average of 4 cm along the core (40 points on the 1mm regular grid)
    d18O_mix = df['d18O_rolling_avg']
    
    ################################################################# VFC[d18O ini & white noise & Mixing] ############################################################
    d18O_mix_noise =  (1.-mixing_level) * d18O_noise_Laepple + mixing_level * d18O_mix
    
    #################################################################### Restore initial mean #################################################################
    d18O_even = d18O_mix_noise
    d18O_even = d18O_even -np.nanmean(d18O_even) + m_d18O;
    
    ############################################################################ Diffusion ############################################################################
    d18O = d18O_even
    depth = depth_even
    Tmean = np.mean(Temp)
    
    d18O[0] = d18O[1]; # Correction for the bad definition of the surface
    depthwe = depth/1000*320 #conversion depth snow non tassee -> en we
    
    depthdummysnow = np.arange(0,1000/320*np.max(depthwe),0.001) #on prepare une profil de la bonne taille entre snow et we
    depthdummywe,rhoddummy =  fm.DensityHL(depthdummysnow,rho,Tmean,accu); #on calcule HL et la profondeur en we
    depthHL = np.interp(depthwe,depthdummywe,depthdummysnow) #conversion we -> en snow equiv
    rhod = np.interp(depthwe,depthdummywe,rhoddummy)

    sigma18,sigmaD = fm.Diffusionlength_OLD(depthHL,rhod,Tmean,650,accu);
    sigma18_storage = (sigma18**(2) + storage_diffusion_cm**(2))**(1/2)   #d'après Dallmayre et al. (2024)
    
    depthHL_regular = np.arange(0,np.max(depthHL),0.001)
    #date_regularHL = np.interp(depthHL_regular, depth_even, date_even)   #pas réussi a régler bug sur date quand je fais ça encore
    d18O_regularHL = np.interp(depthHL_regular, depthHL, d18O)
    sigma18_storage_regularHL = np.interp(depthHL_regular, depthHL, sigma18_storage)
        
    d18O_diff_regularHL = fm.Diffuse_record_OLD(d18O_regularHL,sigma18_storage_regularHL/100,step)[0]  # Rappel: step = 0.001 (grid of 1 mm)
    
    # #POUR SUGBLACIOR STORAGE = 5 YEARS
    #sigma18_15 = (sigma18_19**(2) + 5.6**(2))**(1/2)
    #sigma18 = sigma18_15

    # just added the bit of code Emma sent me by mail
    #d18O_diff = fm.Diffuse_record_OLD(d18O,sigma18/100,step)[0];

    
    #depthHL_regular = np.arange(0,np.max(depthHL),1e-3)
    #d18O_regularHL = np.interp(depthHL_regular,depthHL,d18O)
    #sigma18_regularHL = np.interp(depthHL_regular,depthHL,sigma18)
    #d18O_diff_regularHL = fm.Diffuse_record_OLD(d18O_regularHL,sigma18_regularHL/100,step)[0];

    d18O_diff = np.interp(depthHL,depthHL_regular,d18O_diff_regularHL)

    #block average
    #df = pd.DataFrame({'d18O':d18O_even,'d18O_diff':d18O_diff,'date':pd.DatetimeIndex(date_even)}).set_index(depthHL)
    df = pd.DataFrame({'d18O':d18O_regularHL,'d18O_diff':d18O_diff_regularHL}).set_index(depthHL_regular)   #avec les regular, et sans le Date
    df = df.set_index (df.index + np.diff(df.index)[-1]/2)
    #df_int = block_average(df,.01) # block average at 1cm resolution  
    
    df_int_list = []
    for i in range(len(c_resolution)):
        df_int_i = block_average(df[(df.index >= c_resolution[i][0]) & (df.index < c_resolution[i][1])],c_resolution[i][2])
        df_int_list.append(df_int_i)
    df_int = pd.concat(df_int_list)  
          
    depth_int = df_int.index
    d18O_int = df_int['d18O']
    d18O_diff_int = df_int['d18O_diff']    #NaN
    #date_int = df_int['date']

    return depth_int,d18O_int,d18O_diff_int #,date_int



def VFC_and_spectra_v2(df, core_resolution, storage_diffusion_cm, mix_scale, noise_scale, nl, ml, regular_grid):
    
    spectra_10freq_diff = pd.DataFrame() ; spectra_10psd_diff = pd.DataFrame()
    
    for i_df in range(1,11):
        # VFC
        #depth_int, d18O_int, d18O_diff_int, Date_int_nl0_mv0  = Profile_gen(df.index,df['tsol'],df['precip_adjust']*24*3600,df['precipd18O'], core_resolution, 320, mixing_scale_mm = mix_scale, noise_scale_mm = noise_scale, noise_level=nl, mixing_level=ml)
        depth_int, d18O_int, d18O_diff_int  = Profile_gen(df.index,df['tsol'],df['precip_adjust']*24*3600,df['precipd18O'], core_resolution, storage_diffusion_cm, 320, mixing_scale_mm = mix_scale, noise_scale_mm = noise_scale, noise_level=nl, mixing_level=ml)  #sans le Date
        
        VFC_d18O = pd.DataFrame({'Depth(m)': depth_int, 'd18O no diff': d18O_int, 'd18O diff': d18O_diff_int})
        
        # Interpolate VFC data on a regular grid
        VFC_depth_regular = np.arange(depth_int.min(), depth_int.max(), regular_grid)
        g = 1/regular_grid
        interp_d18O = interp1d(depth_int, d18O_int , kind='linear')
        interp_d18O_diff = interp1d(depth_int, d18O_diff_int, kind='linear')
        VFC_d18O_interp = pd.DataFrame({'Depth(m)': VFC_depth_regular,'d18O no diff': interp_d18O(VFC_depth_regular),'d18O diff': interp_d18O_diff(VFC_depth_regular)})
        
        # Spectrum
        freq, psd = mtm_psd(VFC_d18O_interp['d18O no diff'].dropna(),g)
        spectrum = pd.DataFrame({'freq': freq, 'psd': psd})
        
        freq_diff, psd_diff = mtm_psd(VFC_d18O_interp['d18O diff'].dropna(),g)
        spectrum_diff = pd.DataFrame({'freq_diff': freq_diff, 'psd_diff': psd_diff})
        
        spectra_10freq_diff['freq_diff{}'.format(i_df)] = freq_diff
        spectra_10psd_diff['psd_diff{}'.format(i_df)] = psd_diff
    
    # Average VFC and spectrum diff
    spectra_10freq_diff['freq_diff_mean'] = spectra_10freq_diff.mean(axis=1)
    spectra_10psd_diff['psd_diff_mean'] = spectra_10psd_diff.mean(axis=1)
    
    # Smooth spectrum
    psd_sm, freq_sm = lsm.logsmooth(psd, freq, 0.05)[:2]  #juste le dernier car pas besoin d'en faire plein en non diffusé
    spectrum_sm = pd.DataFrame({'freq_sm': freq_sm, 'psd_sm': psd_sm})
    
    psd_diff_mean_sm, freq_diff_mean_sm = lsm.logsmooth(np.array(spectra_10psd_diff['psd_diff_mean']), np.array(spectra_10freq_diff['freq_diff_mean']), 0.05)[:2]
    spectrum_diff_mean_sm = pd.DataFrame({'freq_diff_mean_sm': freq_diff_mean_sm, 'psd_diff_mean_sm': psd_diff_mean_sm})
    
    # Merge
    spectra = pd.concat([spectrum, spectrum_sm, spectra_10freq_diff, spectra_10psd_diff, spectrum_diff_mean_sm], axis=1)
    
    return VFC_d18O, spectra