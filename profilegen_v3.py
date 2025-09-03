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

from scipy.interpolate import interp1d

def Profile_gen(Date, Temp, Tp, Precip_d18O, Precip_dexc, rho, mixing_level=0, noise_level=0, mixing_scale_m = 40*1e-3, noise_scale_m = 10*1e-3, res = 1e-3,
               storage_diffusion_cm = 0, verbose = False):
    
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
    
    #keep = (Prec>1e-10)
    #Date = Date[keep]
    #Temp = Temp.loc[keep]
    #Prec = Prec.loc[(keep)]
    #if not Precip_d18O is None: Precip_d18O = Precip_d18O.loc[keep]
    
    
    #Generates the profile of isotopic composition of a snow column
    # <!> generates profile at the mm resolution, no block average (see profile_gen_legacy)
    #ddays = np.diff(Date).astype('timedelta64[D]').astype(int) # convert to number of days, and we will assume precip is in units of per days
    #ddays = np.concatenate([[ddays[0]],ddays]) # restore first value, assume it is similar to second value (daily, monthly...)

    #res = 1e-3 # work at the mm resolution for interpolation, diffusion and others...

    reswe = res*rho/1000 # the working resolution for water depth should be smaller by a factor 1000/rho than the target resolution for snow

    ## Data preparation
    
    #accu = np.sum(Tp)/((Date[-1]-Date[0])/pd.Timedelta('365.35D')) # Calculation of the accumulation rate in mm.a-1 w.e.)
    accu = np.sum(Tp)/((Date.iloc[-1]-Date.iloc[0])) # Calculation of the accumulation rate in mm w.e. a-1, assuming Date is in decimal year)

    #Prec = Prec/rho*997/1000 #Conversion to meters and snow equivalent
    #Tp.loc[(Tp<=0)] = 0 #Negative precipitation cannot be taken into account. In the case of true accumulation, where negative values could occur, changes to the script need to be made.
    
    print('accu =', accu)

    ## Compute precipitation intermittent depth serie  

    depth_raw = np.cumsum(Tp.to_numpy()[::-1])[::-1] # sum from present assuming Date is from past to present
    depth_raw = depth_raw/1000  # to meters # ! this is the water equivalent depth in meters !
    #depth_raw -=depth_raw[0]
    #depth_raw_mean = depth_raw[:-1] + np.diff(depth_raw)/2; # middle grid points # I don't use this at the moment

    # Preparing VFC df
    
    vfc = pd.DataFrame({'date':Date.to_numpy(),'d18O_raw':np.array(Precip_d18O),'dexc_raw':np.array(Precip_dexc)},index=depth_raw)
    
    vfc = vfc.sort_index() # place surface depth = 0 at the begining

    ################################################################## Interpolation to a regular grid
    ###############################################################

    # Working on a regular grid in mmwe. This corresponds to uncompacted snow when applying the conversion factor 1000/rho_snow
    # Therefore it is the right depth scale for processes like mixing and white noise which happen on un-compacted snow at the surface

    # J'ai remplacé l'interpolation initiale sur une grille régulière par un bloc average
    # je suspecte qu'une interpolation d'une grille fine sur une grille grossière va beaucoup prendre sur le bruit
    # alors qu'avec un bloc average on est sûr de prendre les valeurs moyennes par segment.
    
    #depthwe_even = np.arange(0,np.max(depth_raw)+reswe,reswe);
    #vfc_even = df_interp(vfc,depthwe_even,kind='next')
    
    #vfc_even = block_average_TEST(vfc,reswe)
    vfc_even = block_average_OLD(vfc,reswe)
    depthwe_even = vfc_even.index.to_numpy()

    ## Sublimation (removing isotopes, no snow layer thickness change, only density)
    ## Condensation from the atmosphere
    ## Frost at the surface
    ## Condensation from the snow below 
    
    ####################################################################### Generate white noise
    #####################################################################
    
    #d18O_ini = vfc_even['d18O_raw'].to_numpy()
    
    #sig_d18O = np.nanstd(d18O_ini); #d18O_int
    #m_d18O = np.nanmean(d18O_ini);
    
    noise_scale_we = noise_scale_m/1000*rho # the imput noise scale in mm SNOW is converted to mm Water with the factor 1000/rho
    noise_scale = int(noise_scale_we/reswe) # converted ot index resolution
    
    # RANDOM AVEC DISTRIBUTION GAUSSIENNE
    #wn18O = np.zeros_like(d18O_ini, dtype=float)
    #for i in range(len(d18O_ini) // noise_scale):
    #    wn18O[noise_scale * i : noise_scale * (i + 1)] = np.random.randn()
    #ind = np.argwhere(np.isnan(wn18O))
    #wn18O[ind]= np.random.randn()

    # same as above but without for loop 
    # should it be the same seed for all species of a different one for Dexc??
    
    f = interp1d(np.arange(0,len(vfc_even)+noise_scale,noise_scale),
                 np.random.normal(0,1,len(np.arange(0,len(vfc_even)+noise_scale,noise_scale))),
                 kind='previous')
    
    wn = f(np.arange(0,len(vfc_even),1)) 

    ################################################################# VFC[d18O ini & white noise] (Laepple)
    ###########################################################

    #dans le code d'emma, factor est calé sur noise_scale = noise_scale_mm en mm de neige
    #pour reproduire ça, on utilise noise_scale_m*100 = noise_scale_mm/10

    # include this factor ??
    #factor = 1/np.sqrt(noise_scale_m*100)
    
    #d18O_noise_Laepple = np.sqrt(1. - noise_level)*d18O_ini + np.sqrt(noise_level)*wnd18O

    vfc_even['d18O_noise'] = np.sqrt(1. - noise_level)*vfc_even['d18O_raw'] + np.sqrt(noise_level)*wn*np.nanstd(vfc_even['d18O_raw']) 
    vfc_even['d18O_noise'] += (1-np.sqrt(1.-noise_level))*np.nanmean(vfc_even['d18O_raw'])
                                                                                                                                                                            
    vfc_even['dexc_noise'] = np.sqrt(1. - noise_level)*vfc_even['dexc_raw'] + np.sqrt(noise_level)*wn*np.nanstd(vfc_even['dexc_raw']) 
    vfc_even['dexc_noise'] += (1-np.sqrt(1.-noise_level))*np.nanmean(vfc_even['dexc_raw'])

    # the series above should by construction have the same std and mean as the original series
    
    ######################################################################### Mixing
    ##################################################################################

    mixing_scale_we = mixing_scale_m/1000*rho # convert input mixing scale in snow depth to water equivalent depth
    mixing_scale = int(mixing_scale_we/reswe) # convert mixing scale to 
    
    vfc_even['d18O_mix'] = vfc_even['d18O_noise'].rolling(window=mixing_scale, center=True,min_periods=1).mean().to_numpy()  # d18O-rolling average of 4 cm along the core (40 points on the 1mm regular grid)

    vfc_even['dexc_mix'] = vfc_even['dexc_noise'].rolling(window=mixing_scale, center=True,min_periods=1).mean().to_numpy()  # d18O-rolling average of 4 cm along the core (40 points on the 1mm regular grid)
    
    ################################################################# VFC[d18O ini & white noise & Mixing]
    ############################################################
    
    d18O_mix_noise =  (1.-mixing_level) * vfc_even['d18O_noise'] + mixing_level * vfc_even['d18O_mix']
    
    dexc_mix_noise =  (1.-mixing_level) * vfc_even['dexc_noise'] + mixing_level * vfc_even['dexc_mix']
    
    vfc_even['d18O'] = d18O_mix_noise.to_numpy()

    vfc_even['dexc'] = dexc_mix_noise.to_numpy()

    vfc_even['dD'] = 8*vfc_even['d18O']+vfc_even['dexc']

    # PHYSICAL PARAMETERS
    ############# Now invoque Herron Langway model to obtain snow depth

    depthsnow_uncompacted = depthwe_even*1000/rho # water depth is converted to uncompacted snow depth with the convertion factor 1000/rho
    # this uncompacted snow depth will overshoot the actual depth
    # this gives us an upperbound on the range over which to compute HerronLangway density profile
    
    Tmean = np.mean(Temp)
    depthwe_HL,rhod =  fm.DensityHL(depthsnow_uncompacted,rho,Tmean,accu); #on calcule HL et la profondeur en we

    vfc_even['rho'] = np.interp(depthwe_even,depthwe_HL,rhod) # we will keep snow density in the dataframe
    
    depthsnow_real = np.interp(depthwe_even,depthwe_HL,depthsnow_uncompacted)

    ############## A vfc with even snowdepth index is now obtained with

    depthsnow_even = np.arange(0,np.max(depthsnow_real),res)
    vfc_even['depth_we'] = vfc_even.index.to_numpy()
    
    vfc_snow_even = df_interp(vfc_even.set_index(depthsnow_real),depthsnow_even)
                                                  
    ############################################################################ Diffusion
    ############################################################################
    # we now apply diffusion to the vfc with in even snow depth
    
    rhod = vfc_snow_even['rho'].to_numpy()
    depthHL = vfc_snow_even.index.to_numpy()
    
    sigma18,sigmaD = fm.Diffusionlength_OLD(depthHL,rhod,Tmean,650,accu);

    #storage_diffusion_cm is an input of profile_gen!!
    #storage_diffusion_cm = 1.5 #for ICORDA
    #storage_diffusion_cm = 5.6 #for subglacior
        
    # #POUR SUGBLACIOR STORAGE = 5 YEARS
    #sigma18_15 = (sigma18_19**(2) + 5.6**(2))**(1/2)
    #sigma18 = sigma18_15
    
    vfc_snow_even['d18O_diff'] = fm.Diffuse_record_OLD(vfc_snow_even['d18O'],sigma18/100,res)[0];

    vfc_snow_even['dD_diff'] = fm.Diffuse_record_OLD(vfc_snow_even['dD'],sigmaD/100,res)[0];

    vfc_snow_even['dexc_diff'] = vfc_snow_even['dD_diff'] - 8 * vfc_snow_even['d18O_diff']


    if storage_diffusion_cm>0:
        sigma18_storage = (sigma18**(2) + storage_diffusion_cm**(2))**(1/2)   #d'après Dallmayre et al. (2024)
        vfc_snow_even['d18O_diff2'] = fm.Diffuse_record_OLD(vfc_snow_even['d18O'],sigma18_storage/100,res)[0];
        vfc_snow_even['sigma18_stored'] = sigma18_storage/100



    vfc_snow_even['sigma18'] = sigma18/100

    vfc_snow_even['sigmaD'] = sigmaD/100

    # format of old output
    #depthHL = vfc_snow_even.index
    #d18O_even = vfc_snow_even['d18O_mix_noise']
    #d18O_even_diff = vfc_snow_even['d18O_diff']
    #date_even = vfc_snow_even['date']

    if verbose is True:
        # keep all working columns (including mixing and noise, and dD)
        outcolumns = vfc_snow_even.columns
    else:
        outcolumns = ['date','d18O_raw','dexc_raw','d18O', 'dexc', 'rho', 'depth_we', 'd18O_diff', 'dexc_diff','sigma18','sigmaD']

    out = vfc_snow_even[outcolumns]
    
    return out



def VFC_and_spectra(df, core_resolution, noise_level, mixing_level, mixing_scale_m, noise_scale_m, vfcres=1e-3, fftres=0.025, repeat = 10):
    
    # res: resolution for the PSD computation. res must be fine enough to capture the resolution of the real core we are comparing with
    
    # vfcres: resolution for the vfc. vfcres must be fine enough to capture mixing and noise scale (1cm)
    
    spectra_10freq_diff = pd.DataFrame() ; spectra_10psd_diff = pd.DataFrame()

    #if nl==0: repeat = 1 # no need to repeat if no random component
        
    for i_df in range(repeat):
        
        # VFC
        
        dfi = Profile_gen(df['decimalyear'],df['tsol'],df['tp_adjust'],df['precipd18O'], 320, 
                          noise_level=noise_level, mixing_level=mixing_level, mixing_scale_m = mixing_scale_m, noise_scale_m = noise_scale_m,
                          res=vfcres)

        # resample with icorda sampling scheme
        #VFC = core_sample_emma(dfi,core_resolution)

        VFC = core_sample_adrien(dfi,core_resolution)

        # put on a regular grid for spectra computation
        VFC_interp = df_interp(VFC,np.arange(min(VFC.index),max(VFC.index),fftres))
        
        # Spectrum

        fs = 1/fftres

        freq, psd = mtm_psd(VFC_interp['d18O'].dropna(),fs)
        spectrum = pd.DataFrame({'freq': freq, 'psd': psd})
        
        freq_diff, psd_diff = mtm_psd(VFC_interp['d18O_diff'].dropna(),fs)
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
      
    return VFC, spectra



