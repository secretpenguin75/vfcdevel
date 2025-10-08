#%% Functions VFC

# generic libraries
from pathlib import Path
import netCDF4
import os
import numpy as np
import xarray as xr
import datetime as datetime
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from inspect import getmembers, isfunction
import pickle as pkl
import scipy
import copy
import matplotlib as mpl
import sys
import scipy

# custom libraries
import vfcdevel.forwardmodel as fm
from vfcdevel.utils import *
from vfcdevel.spectralanalysis import mtm_psd
import vfcdevel.logsmooth as lsm
from vfcdevel.temperature import *

from tqdm.notebook import tqdm

from scipy.interpolate import interp1d


def sublimation_step(deptharray,Marray,isoarray,dspec,tempi,totevapi,subl_depth_m,sublprofile = 'triangle'):

    # dspec = d18O or dD
    # isoarray = isotopes vs depth in mmwe at step i
    # res resolution in mmwe of the corresponding depth array
    # tempi = surface temperature at step i
    # evapi = evaporation at step i

    #totevapi = Evap.iloc[i]
    #tempi = Temp.iloc[i]
    
    spec = dspec[1:] # remove d prefix

    # prepare fractionation coefficient in dictionaries and arrays

    #rhowater = 1000

    #M = reswe*rhowater # mass in a layer of 1mm (we are still in water equivalent)

    #M = np.diff(deptharray,append=np.nan)*rhoarray # mass in each layer / m2 = dz*rho
    
    M = Marray # new! we keep track of the mass anomaly in each layer
    
    #h = df['h'] # varying saturatio1
    h=0.8
    
    k = fm.kdiff('Cappa',spec,0.4)
    
    alphaeq = fm.Frac_eq('Maj', spec ,tempi-273.15) # considering surface temperature only

    # create a 1D array that has just the sublimation values at time Ti

    if sublprofile =='triangle':
        weight = np.maximum(subl_depth_m-deptharray,0)
        weight /= np.sum(weight)
    elif sublprofile == 'square':
        weight = (deptharray<subl_depth_m)
        weight /= np.sum(weight)
    elif sublprofile =='exponential':
        weight = np.exp(-deptharray/subl_depth_m)
        weight /= np.sum(weight)
    
    TE = weight*totevapi 

    sigma = (M - 1/alphaeq*(1-k)/(1-h*k)*TE)/(M-TE) # <----- a vector that works spatially

    if np.max(TE>M)>0:
        print('warning, evaporation exceeded snow layer content')

    isoarraysubl = sigma*isoarray+(sigma-1)*1e3 # one operation sublimation

    return isoarraysubl,TE
    
def Profile_gen(Date, Temp, Tp, Proxies, rho, Te = None, mixing_level=0, noise_level=0, mixing_scale_m = 40*1e-3, noise_scale_m = 10*1e-3, res = 1e-3,
               storage_diffusion_cm = None , verbose = True, keeplog = False, logperiod = 1,subl_depth_m = 50*1e-3):


    reswe = res*rho/1000 # the working resolution for water depth should be smaller by a factor 1000/rho than the target resolution for snow

    ## Data preparation
    
    #accu = np.sum(Tp)/((Date[-1]-Date[0])/pd.Timedelta('365.35D')) # Calculation of the accumulation rate in mm.a-1 w.e.)
    accu = np.sum(Tp)/((Date.iloc[-1]-Date.iloc[0])) # Calculation of the accumulation rate in mm w.e. a-1, assuming Date is in decimal year)

    #Prec = Prec/rho*997/1000 #Conversion to meters and snow equivalent
    #Tp.loc[(Tp<=0)] = 0 #Negative precipitation cannot be taken into account. In the case of true accumulation, where negative values could occur, changes to the script need to be made.
    
    if verbose: print('accu =', accu)

    ## Compute precipitation intermittent depth serie  

    depthwe_raw = Tp.iloc[::-1].cumsum().to_numpy()[::-1]
    depthwe_raw = depthwe_raw/1000

    # Preparing VFC df

    # Proxies can be either a pd.Series (df_lmdz['precipd18O']) OR a dataframe (df_lmdz[['precipd18O','tsol_d18O','tsol_dD',...]])
    # Proxies contains columns for d18O, dD (support for 17O to be added later)
    # in order to identify which species is represented (for diffusion) we keep track of it using Pandas Dataframe attrs (attributes)
    # the Proxies.attrs['species'] dictionary as an isotopic species (d18O or dexc or dD) associated to each columns in Proxies

    # handle the case when storage diffusion cm is a float
    if type(storage_diffusion_cm) in [int,float]:
        storage_diffusion_cm = {spec:storage_diffusion_cm for spec in ['d18O','dD']}

    
    if type(Proxies) == pd.Series:
        Proxies = pd.DataFrame(Proxies)

    speciesmap = read_species(Proxies)
    if verbose:
        print('Species assigned automatically as:')
        print(speciesmap)
    species = list(speciesmap.values())
    
    Proxies = Proxies.rename(columns = speciesmap)
    
    if ('d18O' in species and 'dexc' in species) and 'dD' not in species:
        # if d18O and dexc are given as input, add dD as it is on dD that we apply subl,diffusion, etc...
        Proxies['dD'] = 8*Proxies['d18O']+Proxies['dexc']
        species += ['dD']
    

    # keep track of deposition time
    Proxies.insert(0,'deposition_time',Date)


    # INITIAL INTERPOLATION STEP
    # here we interpolate the initial data on depth_raw axis to a depth_even axis
    # in order to respect the initial precipitation weights, we first interpolate on a finer grid with 'next' method
    # then we apply a block average, these two steps ensure precipitation weighting
    
    subsampling = 100

    depth_hires = np.arange(0,max(depthwe_raw),reswe/subsampling)

    ff = scipy.interpolate.interp1d(depthwe_raw[::-1],Proxies.to_numpy()[::-1],kind='next',fill_value='extrapolate',axis=0)

    proxies_hires = ff(depth_hires)

    depth_even_bins = np.arange(0,max(depthwe_raw)+reswe,reswe)

    proxies_even = scipy.stats.binned_statistic(depth_hires,proxies_hires.T,bins=depth_even_bins)[0].T
    depthwe_even = depth_even_bins[:-1]

    xvfc0 = pd.DataFrame(proxies_even,index=pd.Index(depthwe_even,name='depth'),columns = Proxies.columns).to_xarray()
    
    # END OF INITIAL INTERPOLATION STEP

    

    # Prepare physical parameters: density
    
    #depthwe = xvfc0_we['depth'].values
    Tmean = np.nanmean(Temp)
    depthsnow,rhod = fm.DensityHL_we(depthwe_even,rho,Tmean,accu)
    
    
    xvfc0['depthwe'] = xr.DataArray(depthwe_even,coords={'depth':xvfc0.coords['depth']}) # -> redundant for now with even depthwe axis, will serve as a trace of depthwe later
    xvfc0['rhoHL'] = xr.DataArray(rhod,coords={'depth':xvfc0.coords['depth']})
    xvfc0['depthsnow'] = xr.DataArray(depthsnow,coords={'depth':xvfc0.coords['depth']}) # -> physical depth to use in time loop
    
    xvfc0['total_mass'] = xr.DataArray(np.full(depthwe_even.shape,reswe*1000),coords={'depth':xvfc0.coords['depth']}) # new! keep track of the water mass in each column
    
    xvfc0 = xvfc0.rename({key:key+'_raw' for key in species}) # change column names to 'd18O_raw' and 'dexc_raw'    
    
    for dspec in list(set(['d18O','dD']) & set(species)):
        # duplicate isotope column for later analyses with and without post dep effect
        xvfc0[dspec] = xvfc0[dspec+'_raw'].copy()    

    #depth_we = xvfc0['depthwe'].values
    time_even = xvfc0['deposition_time'].values


    # depth_we: uniform array at resolution reswe with max depthwe of final core
    # depth_even: uniform array at resolution res with max snow depth of final core
    # we will perform we (add new snow) operations on depth_we
    # and snow operations on depth_even

    # CONTINUE HERE
    # situation for now: keep dataset in depthwe even, use dephtsnow for computations

    
    if keeplog is False and Te is None:

        # if we do not want to keep the time log (for example to monitor surface snow)
        # and if surface processes are off 
        # then we can skip the timeloop alltogether and compute VFC as in the first versions

        
        # INTRODUCE VFC IN REAL SNOW DEPTH
        xvfc = copy.deepcopy(xvfc0)

        # introduce a dummy, length 0 dimension for time in case there is no need for time loop

        coolvars = ['deposition_time','d18O','dD','d18O_raw','dD_raw','total_mass']
        xvfc[coolvars] = xvfc[coolvars].expand_dims({"time":[Date.values[-1]]},axis=0).isel(time=-1)

    else:

        # we add a column for firn temperature if time loop is running
        
        xvfc0['tfirn'] = np.nan

        
        if Te is not None:
            # add a column to keep track of evaporation (this could be dropped in a final version)
            Te = np.maximum(Te,0) # for now, ignore negative sublimation
            xvfc0['te'] = np.nan


        # if history in on AND/OR for post deposition INTRODUCE THE TIME coordinate for the time loop
        
        # this will be overwritten but we generate an empty version to get dimensions right

        xvfc = copy.deepcopy(xvfc0)
        

        # variables considered for the time loop
        coolvars = ['deposition_time','d18O','dD','d18O_raw','dD_raw','tfirn','te','total_mass']
        
        for key in coolvars:
            xvfc[key] = xr.full_like(xvfc[key],np.nan)
                
        xvfc[coolvars] = xvfc[coolvars].expand_dims({"time":Date.values},axis=0) # only add a time axis to the quantities that evolve over time

        # we now transition to numpy arrays which will make the time loop operations much faster
        vfc0 = xvfc0[coolvars].to_dataarray().transpose('depth','variable').values
    
        xvfc_da = xvfc[coolvars].to_dataarray().transpose('time','depth','variable') # bring time as first variable for convenience
        # we will call xvfc_da once more later to retrieve dimension structure
        # but for now, switch to numpy array
        
        logvfc = xvfc_da.values # numpy array for faster time loop operations
    
        # TEMPERATURE parameters
        D = 19/24/3600/365.25 # temperature diffusivity in firn 19 meter per year
        dt = np.diff(Date)[0]*365.25*24*3600 # timestep in decimalyear to seconds
        Tdepthmax = 4*np.sqrt(D*24*3600*365.25) // 1 + 1.01#  = 4* skin depth of yearly cycle
        Tdepth = np.arange(0,Tdepthmax,1e-2) # maximum depth, in cm resolution, on which to compute temperature profile
        cc = [np.arange(0,len(Tdepth),25)] # for now we will solve the temperature loop at 25cm resolution; the penetration depth of the daily cycle
        Ti = np.full((Tdepth).shape,np.nanmean(Temp))

        # TIME LOOP FOR VAPOUR PROCESSES
        timebar = tqdm(range(len(Date)),'Processing time loop',colour='green')

        # now that we have switched to numpy, keep track of column indices with;
        indM = coolvars.index('total_mass')
        indE = coolvars.index('te')
        indT = coolvars.index('tfirn')
        
        for i in timebar:
            #### in depth

            # determine how much new snow is to be added in the current step
            if i == 0:
                maski = (time_even<=Date.iloc[0]) # layer beetween snow deposited at time t_i and snow deposited at time t_i+1
            elif i==len(Date)-1:
                # weird numerical bug where binned times (average over reswe) is higher than max time (diff = 5e-12), while hires never is.
                # to avoid missing a bit we slightly adjust the mask condition for the last step
                maski = (time_even>Date.iloc[i-1])
            else:
                maski = (time_even<=Date.iloc[i])&(time_even>Date.iloc[i-1]) # layer beetween snow deposited at time t_i and snow deposited at time t_i+1    

            if sum(maski)>0:
                logvfci = np.full(vfc0.shape,np.nan)
                # shift by amount of new snow in water equiv ( does nothing if sum(maski0)=0 )
                logvfci[sum(maski):] = logvfc[i-1][:-sum(maski)]
                # feed new values
                logvfci[:sum(maski)] = vfc0[maski]

            else:
                logvfci = logvfc[i-1]

            # update temperature profile
            Ti = temperature_step(Ti,Tdepth,Temp.iloc[i],D,dt,cc) # temperature step computed on an independant depth axis (<epaisseur de peau)
            # that might overshoot or on the contrary undershoot the actual depth
            #ff = scipy.interpolate.interp1d(Tdepth,Ti,kind='linear',bounds_error='extrapolate')
            logvfci[:,indT] = np.interp(depthsnow,Tdepth,Ti) # linear interpolation of real grid
        
            # sublimation comes here
    
            #### SUBLIMATION STARTS HERE
            # takes the previous iteration of isotope array and applies
            # the linear transformation packed in sublimationstep
        
            if Te is not None:
                for dspec in list(set(['d18O','dD']) & set(species)):
                    ind = coolvars.index(dspec)
                    
                    #subl_depth_we = subl_depth_m*rho/1000
                    logvfci[:,ind],TEi = sublimation_step(depthsnow,logvfci[:,indM],logvfci[:,ind],dspec,Temp.iloc[i],Te.iloc[i],subl_depth_m,sublprofile = 'exponential')
                logvfci[:,indE] = TEi
                logvfci[:,indM] -= TEi
                    
            ###### condensation comes here
            # paving the way for condensation: compute Cv vapour content with temperature profile

            

            
            logvfc[i] = copy.deepcopy(logvfci)
    
        # when we are done with the loop, convert numpy array to xr dataset
        
        coords = xvfc_da.coords
        
        xvfc[coolvars] = xr.DataArray(logvfc,coords=coords).to_dataset(dim='variable') # clumsy but I am a xr noobie okay
        
        #########################################################################
        #########################################################################
        ########################################################### CONTINUE HERE
        
        # only keep one record every ::logperiod for faster processing of diffusion

        if keeplog is False:
            
            xvfc = xvfc.isel(time=-1) #only keep the last iteration

        else:
            # logperiod tells the frequency at which to keep a trace of the vfc
            # in future version, could be a slice of df['date'] etc
            
            xvfc = xvfc.isel(time=slice(None,None,-1)).isel(time=slice(None,None,logperiod)).isel(time=slice(None,None,-1))

    # time to switch from even water depth array to even depth array in snow depth


    #xvfcold = copy.deepcopy(xvfc)
    
    xvfc.coords['depth'] = depthsnow

    # a bit of extra care for accumulated quantities
    # going from even mmwe depth axis to even snow depth axis, we conserve mass fraction

    # WARNING: it seems like we are loosing a bit of mass (and no evap?) with this method
    # will have to be refined

    if Te is not None:
        xvfc['te_acc'] = xvfc['te'].cumsum(dim='depth')
        xvfc['total_mass_acc'] = xvfc['total_mass'].cumsum(dim='depth')

    xvfc = xvfc.interp({'depth':np.arange(0,np.max(depthsnow),res)},method='linear')

    if Te is not None:
        xvfc['te'] = xvfc['te_acc'].diff(dim='depth')
        xvfc['te'].transpose('depth',...).loc[0] = xvfc['te_acc'].sel(depth=0).values


        xvfc['total_mass'] = xvfc['total_mass_acc'].diff(dim='depth')
        xvfc['total_mass'].transpose('depth',...).loc[0] = xvfc['total_mass_acc'].sel(depth=0).values


        del xvfc['te_acc']
        del xvfc['total_mass_acc']
    
    #xvfc['mass_anomaly'] = xvfc['total_mass']/xvfc0['total_mass'].values

    sigma18,sigmaD = fm.Diffusionlength_OLD(xvfc['depth'].values,xvfc['rhoHL'].values,Tmean,650.,accu);

    xvfc['sigma18'] = xr.DataArray(sigma18,coords={'depth':xvfc.coords['depth']})
    xvfc['sigmaD'] = xr.DataArray(sigmaD,coords={'depth':xvfc.coords['depth']})


    specbar = tqdm([('d18O','sigma18'),('dD','sigmaD')],'Applying diffusion',colour='green',leave = verbose)
    
    #for speci,sigmai in [('d18O','sigma18'),('dD','sigmaD')]:
    for speci,sigmai in specbar:
        
        for column in xvfc.var(): # scan variables and assign species one more time
            if speci in column:
                
                # with this matrix approach I am still figuring out a way to treat the surface boundary
                # (with nans on top) so the diffusion weighting ignores the nan
                # for now do fillna(0) to actually get a result and restore nans with .where()
                # however since sigma is less than 1mm up to 5mm depth the error is really small
                # but we need to control it to try and avoid any artifact in surface snow
                # work in progress
                
                xvfc[column+'_diff'] = xr.DataArray(fm.Diffuse_record(xvfc[column].fillna(0).transpose(...,'depth').values,xvfc[sigmai].values/100/res),
                                                    coords=xvfc[column].transpose(...,'depth').coords).where(np.isfinite(xvfc[column]))
            
    #     if spec in storage_diffusion_cm.keys():
    #         sigma_storage_cm = (sigma_cm[spec]**(2) + storage_diffusion_cm[spec]**(2))**(1/2)   #d'après Dallmayre et al. (2024)
    #         vfc_snow_even[spec+'_diff2'] = fm.Diffuse_record_OLD(vfc_snow_even[spec],sigma_storage_cm/100,res)[0];
    #         vfc_snow_even['sigma'+spec+'_stored'] = sigma_storage_cm/100

    # compute dexc if it was part of the input
    
    if 'dD_diff' in list(xvfc.var()) and 'd18O_diff' in list(xvfc.var()):
        xvfc['dexc'] = xvfc['dD'] - 8 * xvfc['d18O']
        xvfc['dexc_raw'] = xvfc['dD_raw'] - 8 * xvfc['d18O_raw'] # overwrites with the timexdepth dataarray
        xvfc['dexc_diff'] = xvfc['dD_diff'] - 8 * xvfc['d18O_diff']
        xvfc['dexc_raw_diff'] = xvfc['dD_raw_diff'] - 8 * xvfc['d18O_raw_diff']

    return xvfc
