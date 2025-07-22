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

def Profile_gen(Date,Tp,Iso,Temp,rho, noise_scale_we = 20 ,noise_level = 0.8):
        
    #Generates the profile of isotopic composition of a snow column

    # Assume Total precipitation is in units of mm w.e, or kg/m^2

    # Assume Date is in Decimal year

    # <!> generates profile at the mm resolution, no block average (see profile_gen_legacy)



    ## Data preparation
        
    #accu = np.sum(Tp)/((Date[-1]-Date[0])/pd.Timedelta('365.25D')) # Calculation of the accumulation rate in mm.a-1 w.e.)
    accu = np.sum(Tp)/(Date.iloc[-1]-Date.iloc[0]) # Calculation of the accumulation rate in mm.a-1 w.e.)

    print(accu)


    Tp.loc[(Tp<=0)] = 0 #Negative precipitation cannot be taken into account.
                                   #In the case of true accumulation, where negative 
                                    #values could occur, changes to the script need to be made.
    
    ## Compute precipitation intermittent depth serie

    depthwe_raw = np.cumsum((Tp.to_numpy())[::-1])[::-1] # sum from the surface
    depthwe_raw *=1/1000 # conversion from mm to meters
    depthwe_raw = depthwe_raw - depthwe_raw[-1] # set surface depth to zero

    iso_raw = Iso.to_numpy()

    date_raw = Date.to_numpy()
    
    #temp_raw = Temp.to_numpy()

        

    #depth_raw_mean = depth_raw[:-1] + np.diff(depth_raw)/2; # middle grid points # I don't use this at the moment

    # Sublimation (removing isotopes, no snow layer thickness change, only density)

    

    # Condensation from the atmosphere
    # Frost at the surface

    # Condensation from the snow below

    ## Interpolation to a regular grid
    res = 1e-3; #in meters, so mm resolution, for large accumulation or very long time series, this can lead to large variables.
    
    maxdepth = depthwe_raw[0]
    depthwe_even = np.arange(0,maxdepth,res);
    iso_even = np.interp(depthwe_even,depthwe_raw[::-1],iso_raw[::-1]);
    #Dating = np.interp(depth,depth_raw_mean,datenum(date_raw));
    #date_even = float_time_interp(depthwe_even,depthwe_raw[::-1],Date[::-1])

    date_even = np.interp(depthwe_even,depthwe_raw[::-1],date_raw[::-1])
    #temp_even = np.interp(depth_even,depth_raw[::-1],temp_raw[::-1])
    

    ## Generating a white noise
    ## Here we work in mmwe depth, by simply scaling mixing and noise windows with density
    ## this is justified by the fact that there is no compression in the upper layers
    
    std_iso = np.nanstd(iso_even);
    mean_iso = np.nanmean(iso_even);

    wn_iso = np.full(iso_even.shape,np.nan)

    noise_scale = int(noise_scale_we/1000/res) # compute noise scale from mmwe to resolution unit
    
    for i in range(len(iso_even)//noise_scale):
        # same wn value for per block of noise scale, right?
        wn_iso[noise_scale*i:noise_scale*(i+1)] = np.random.normal(0,1)
    wn_iso[noise_scale*(i+1):] = np.random.normal(0,1)

    iso_even = ((1 - noise_level))**(1/2)*iso_even + (noise_level)**(1/2)*std_iso*wn_iso; # std of iso_even is conserved with this formula (std_1+2 = sqrt( std1**2+std2**2)
    
    iso_even = iso_even -np.nanmean(iso_even) + mean_iso; # restore initial mean


    ## SNOW MIXING

    

    ## DIFFUSION

    
    Tmean = np.mean(Temp)
    
    depthsnow0 = depthwe_even*1000/rho #this uncompacted snow depth will overshoot the actual depth of the core
    
    depthwe0,rhod0 =  fm.DensityHL(depthsnow0,rho,Tmean,accu); #on calcule Herron and Langway et la profondeur en we

    #depthHL is the snow depth of the original mmwe scale according to the Herron and Langway model
    
    depthHL = np.interp(depthwe_even,depthwe0,depthsnow0) #conversion wwater equiv. -> en snow equiv.
    rhod = np.interp(depthwe_even,depthwe0,rhod0)

    Pressure = 650 # Pressure at Dome C, to be relaxed as a free parameter later on
    sigma18,sigmaD = fm.Diffusionlength_OLD(depthHL,rhod,Tmean,Pressure,accu);
    sigma18_e = sigma18/100/res# sigma18 is in cm, we must restore to meters and then to resolution
    iso_even_diff = fm.Diffuse_record(iso_even,sigma18_e); 


    return depthHL,depthwe_even,rhod,iso_even,iso_even_diff,date_even,sigma18

def Profile_gen_legacy(Date,Prec,Iso,Temp,rho,noise_scale = 20,noise_level = 0.8,sampling_res=0.015):

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
    Prec = Prec.loc[keep]
    Iso = Iso.loc[keep]
    
    # generate core
    
    depth,depthwe,rhod,iso,iso_diff,date = Profile_gen(Date,Prec,Iso,Temp,rho,noise_scale,noise_level)

    # block average at sampling_res resolution

    df = pd.DataFrame({'iso':iso,'iso_diff':iso_diff,'date':pd.DatetimeIndex(date)}).set_index(depth)
    df_int = block_average(df,sampling_res) # block average at 1cm resolution
    #df_int = block_average(df,sampling_res) # block average at 3cm resolution

    depth_int = df_int.index
    iso_int = df_int['iso']
    iso_diff_int = df_int['iso_diff']
    date_int = df_int['date']

    return depth_int,iso_int,iso_diff_int,date_int