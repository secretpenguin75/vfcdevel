from vfcdevel.utils import time_float_interp
import numpy as np
import scipy
import matplotlib.pyplot as plt

def model_to_obs_fit(model,obs):
    
    # if model is in hourly time step, and obs in daily
    # make sure to average the model to daily timesteps first
    
    x = time_float_interp(obs.index,model.index,model.values)
    y = obs.values
    ind = np.logical_and(np.isfinite(x),np.isfinite(y))
    a,b = scipy.stats.linregress(x[ind],y[ind])[:2]
    plt.scatter(x,y,color='blue')
    plt.plot(x,a*x+b,color='red')
    return a,b

def temp_to_iso_DC(Temp,iso,fit='era5'):

    # input: Temperature in Kelvin

    #Definition of the isotopic composition - temperature relationship
    if fit=='Mathieu':
        alpha = 0.46; 
        beta = -32;
        
    if fit=='era5':
        alpha = 0.49
        beta=-31

    if fit=='lmdz':
        alpha = 0.38
        beta=-33
    
    d18O = alpha*(Temp-273.15)+beta

    #dexc = -0.68*d18O-25;
    #dexc = -1.04*d18O-43
    dexc = -0.5*d18O-15.7 # Mathieu's fit
    
    if iso=='18O':
        out = d18O

    if iso=='D':
        out = dexc+8*d18O

    if iso=='d17':
        O17exc = 1.5*d18O + 103;
        out = O17exc

    if iso=='dexc':
        out = dexc
        
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