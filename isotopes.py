from vfcdevel.utils import time_float_interp
import numpy as np
import scipy
import matplotlib.pyplot as plt

def model_to_obs_fit(model,obs):
    
    # if model is in hourly time step, and obs in daily
    # make sure to average the model to daily timesteps first
    
    x = time_float_interp(obs.index,model.index,model.values,left=np.nan,right=np.nan)
    y = obs.values
    ind = np.logical_and(np.isfinite(x),np.isfinite(y))
    a,b = scipy.stats.linregress(x[ind],y[ind])[:2]
    plt.scatter(x,y,color='blue')
    plt.plot(x,a*x+b,color='red')
    return a,b

def temp_to_iso_DC(Temp,iso,fit='era5',dfit='default'):

    # input: Temperature in Kelvin

    #Definition of the isotopic composition - temperature relationship
    
    if fit=='Mathieu':
        # equation (1) in casado 2021
        alpha = 0.46; 
        beta = -32;
        
    if fit=='era5 t2m':
        # fit of era t2m on the period 2008-2022 ignoring precip samples with dexc<0
        alpha = 0.44
        beta=-33

    if fit=='lmdz tsol':
        # old fit based on lmdz tsol
        alpha = 0.33
        beta=-37

    if fit=='lmdz t2m':
        alpha = 0.36
        beta=-36
    
    d18O = alpha*(Temp-273.15)+beta

    # from surf_cycle_gen.m
    # d18O_s = 0.46*Temp -32;
    # dexc_s = -0.68*d18O_s-25; # this is the value for "annual weighted' in Dreossi 2024
    # O17exc_s = 1.5*d18O_s + 103;
    
    #dexc = -0.5*d18O-15.7 # Mathieu's fit (excerpt from code, couldnt find paper citation)
    
    #dexc = -1.35 * d18O - 61 # Dreossi 2024 (2008-2017)

    if dfit =='weighted':
        
        dexc = -0.68 * d18O - 25 # this is the value for "annual weighted' in Dreossi 2024

    if dfit == 'default':
        
        dexc = -1.04 * d18O - 43 # my fit with dexc > 0
    
    if iso=='18O':
        out = d18O

    if iso=='dexc':
        out = dexc

    if iso=='D':
        out = dexc+8*d18O

    if iso=='d17':
        O17exc = 1.5*d18O + 103;
        out = O17exc


        
    return out

def R_to_delta(R,iso):
    
    R_VSMOW = {'18O' : 2005.2*10**(-6),
               '17O' : 379.9*10**(-6),
               'D'  : 155.76*10**(-6),
              }

    delta = (R/R_VSMOW[iso]-1)*1000
    
    return delta

def delta_to_R(delta,iso):
    
    R_VSMOW = {'18O' : 2005.2*10**(-6),
               '17O' : 379.9*10**(-6),
               'D'  : 155.76*10**(-6),
              }

    #delta = (R/R_VSMOW[iso]-1)*1000
    R = (delta/1000+1)*R_VSMOW[iso]
    
    return R