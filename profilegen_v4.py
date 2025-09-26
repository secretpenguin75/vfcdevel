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
from tqdm.notebook import tqdm

from scipy.interpolate import interp1d


def sublimation_step(isoarray,dspec,tempi,totevapi,subl_depth_we,reswe,sublprofile = 'triangle'):

    # dspec = d18O or dD
    # isoarray = isotopes vs depth in mmwe at step i
    # res resolution in mmwe of the corresponding depth array
    # tempi = surface temperature at step i
    # evapi = evaporation at step i

    #totevapi = Evap.iloc[i]
    #tempi = Temp.iloc[i]
    
    spec = dspec[1:] # remove d prefix

    # prepare fractionation coefficient in dictionaries and arrays

    rhowater = 1000

    M = reswe*rhowater # mass in a layer of 1mm (we are still in water equivalent)

    #h = df['h'] # varying saturation
    h=0.8
    
    k = fm.kdiff('Cappa',spec,0.4)
    alphaeq = fm.Frac_eq('Maj', spec ,tempi-273.15) # considering surface temperature only

    #subl_depth_we = subl_depth_m/1000
    subl_depth_e = int(subl_depth_we/reswe)

    # create a 1D array that has just the sublimation values at time Ti

    if sublprofile =='triangle':
        TE = np.full(len(isoarray),0.) # a 1-D array that has the same length as depth axis
        TE[:subl_depth_e] = np.arange(subl_depth_e)[::-1]/np.sum(np.arange(subl_depth_e))*totevapi # triangle shape        
    elif sublprofile == 'square':
        TE = np.full(len(isoarray),0.) # a 1-D array that has the same length as depth axis
        TE[:subl_depth_e] = totevapi/subl_depth_e  # here square shape of length k
    elif sublprofile =='exponential':
        TE = np.exp(-np.arange(len(isoarray))/subl_depth_e)/np.sum(np.exp(-np.arange(len(isoarray))/subl_depth_e))*totevapi # exponential        

    sigma = (M - 1/alphaeq*(1-k)/(1-h*k)*TE)/(M-TE) # <----- a vector that works spatially

    if max(TE)>M:
        print('warning')

    isoarraysubl = sigma*isoarray+(sigma-1)*1e3 # one operation sublimation, with mean temperature at surface

    return isoarraysubl,TE

def Profile_gen(Date, Temp, Tp, Proxies, rho, Te = None, mixing_level=0, noise_level=0, mixing_scale_m = 40*1e-3, noise_scale_m = 10*1e-3, res = 1e-3,
               storage_diffusion_cm = None , verbose = False, keeplog = False, logperiod = 1,subl_depth_m = 50*1e-3):


    reswe = res*rho/1000 # the working resolution for water depth should be smaller by a factor 1000/rho than the target resolution for snow

    ## Data preparation
    
    #accu = np.sum(Tp)/((Date[-1]-Date[0])/pd.Timedelta('365.35D')) # Calculation of the accumulation rate in mm.a-1 w.e.)
    accu = np.sum(Tp)/((Date.iloc[-1]-Date.iloc[0])) # Calculation of the accumulation rate in mm w.e. a-1, assuming Date is in decimal year)

    #Prec = Prec/rho*997/1000 #Conversion to meters and snow equivalent
    #Tp.loc[(Tp<=0)] = 0 #Negative precipitation cannot be taken into account. In the case of true accumulation, where negative values could occur, changes to the script need to be made.
    
    print('accu =', accu)
    print('tmean =',np.nanmean(Temp))

    ## Compute precipitation intermittent depth serie  

    depth_raw = Tp.iloc[::-1].cumsum().to_numpy()[::-1]
    depth_raw = depth_raw/1000

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
    #print('Species assigned automatically as:')
    print(speciesmap)
    species = list(speciesmap.values())
    
    Proxies = Proxies.rename(columns = speciesmap)
    
    if ('d18O' in species and 'dexc' in species) and 'dD' not in species:
        # if d18O and dexc are given as input, add dD as it is on dD that we apply subl,diffusion, etc...
        Proxies['dD'] = 8*Proxies['d18O']+Proxies['dexc']
        species += ['dD']

    #proxies_dic = {key:value for key,value in Proxies.items()}


    ##############################################
    ##############################################

    # Here initialize temperature profiles that will be used in condensation computation
    # and leave an option to re--use them for later computation
    # temperature profiles to be added as (time) x (depth) array in xvfc

    ##############################################
    ##############################################
    

    # keep track of deposition time
    Proxies.insert(0,'deposition_time',Date)


    # INITIAL INTERPOLATION STEP
    # here we interpolate the initial data on depth_raw axis to a depth_even axis
    # in order to respect the initial precipitation weights, we first interpolate on a finer grid with 'next' method
    # then we apply a block average, these two steps effectively preserve precipitation weighting
    
    subsampling = 100

    depth_hires = np.arange(0,max(depth_raw),reswe/subsampling)

    ff = scipy.interpolate.interp1d(depth_raw[::-1],Proxies.to_numpy()[::-1],kind='next',fill_value='extrapolate',axis=0)

    proxies_hires = ff(depth_hires)

    depth_even_bins = np.arange(0,max(depth_raw)+reswe,reswe)

    print(proxies_hires.shape)

    proxies_even = scipy.stats.binned_statistic(depth_hires,proxies_hires.T,bins=depth_even_bins)[0].T
    depth_even = depth_even_bins[:-1]

    xvfc0 = pd.DataFrame(proxies_even,index=pd.Index(depth_even,name='depth'),columns = Proxies.columns).to_xarray()

    xvfc0 = xvfc0.rename({key:key+'_raw' for key in species}) # change column names to 'd18O_raw' and 'dexc_raw'    

    # END OF INITIAL INTERPOLATION STEP

    
    colind = {}
    for dspec in list(set(['d18O','dD']) & set(species)):
        # duplicate isotope column for later analyses with and without post dep effect
        xvfc0[dspec] = xvfc0[dspec+'_raw'].copy()
        # keep track of column index for the time loop
        colind[dspec] = list(xvfc0.to_dataarray()['variable']).index(dspec)

    # this will be overwritten but we generate an empty version to get dimensions right

    #if keeplog is False and Te is None:


    
    if keeplog is False and Te is None:

        # if we do not want to keep the time log (for example to monitor surface snow)
        # and if surface processes are off 
        # then we can skip the timeloop alltogether and compute VFC as in the first versions

        # introduce a dummy, length 0 dimension for time in case there is no need for time loop
        
        xvfc_we = copy.deepcopy(xvfc0.expand_dims({"time":[Date.values[-1]]},axis=0).isel(time=-1))

    else:

        
        if Te is not None:
            # add a column to keep track of evaporation (this could be dropped in a final version)
            Te = np.maximum(Te,0) # for now, ignore negative sublimation
            xvfc0['evap'] = np.nan
            colind['evap'] = list(xvfc0.to_dataarray()['variable']).index('evap')


        # if history in on AND/OR for post deposition INTRODUCE THE TIME coordinate for the time loop
        xvfc_we = copy.deepcopy(xr.full_like(xvfc0,np.nan).expand_dims({"time":Date.values},axis=0))
    
        vfc0 = xvfc0.to_dataarray().transpose('depth','variable').values
    
        xvfc_we_da = xvfc_we.to_dataarray().transpose('time','depth','variable')# bring time as first variable for convenience
        
        logvfc = xvfc_we_da.values # numpy array for faster time loop operations
    
    
        # TIME LOOP FOR VAPOUR PROCESSES
        timebar = tqdm(range(len(Date)),'Processing time loop',colour='green')

        
        for i in timebar:
            #### in depth
            logvfci = np.full(vfc0.shape,np.nan)
                
            if i<len(timebar)-1:
        
                # feed new values
                maski0 = (depth_even<=depth_raw[i])&(depth_even>depth_raw[i+1]) # layer beetween snow deposited at time t_i and snow deposited at time t_i+1
                logvfci[:sum(maski0)] = vfc0[maski0] # position of profile_i on the final profile
        
                # feed previous timestep
                maski = (depth_even>depth_raw[i]) # deeper than layer i
                logvfci[sum(maski0):sum(maski0)+sum(maski)] = logvfc[i-1][:sum(maski)]
    
            if i == len(timebar)-1:
        
                # feed new values
                maski0 = (depth_even<=depth_raw[i]) # layer beetween snow deposited at time t_i and snow deposited at time t_i-1
                logvfci[:sum(maski0)] = vfc0[maski0] # position of profile_i on the final profile
        
                # feed previous timestep
                maski = (depth_even>depth_raw[i]) # deeper than layer i
                logvfci[sum(maski0):sum(maski0)+sum(maski)] = logvfc[i-1][:sum(maski)]
        
            # sublimation comes here
    
            #### SUBLIMATION STARTS HERE
            # takes the previous iteration of isotope array and applies
            # the linear transformation packed in sublimationstep
        
            if Te is not None:
                for dspec in list(set(['d18O','dD']) & set(species)):
                    ind = colind[dspec]
                    subl_depth_we = subl_depth_m*rho/1000
                    logvfci[:,ind],TEi = sublimation_step(logvfci[:,ind],dspec,Temp.iloc[i],Te.iloc[i],subl_depth_we,reswe,sublprofile = 'exponential')
                logvfci[:,colind['evap']] = TEi
                    
                # condensation comes here
        
            logvfc[i] = copy.deepcopy(logvfci)
    
        # when we are done with the loop, convert numpy array to xr dataset
        
        coords = xvfc_we_da.coords
        xvfc_we = xr.DataArray(logvfc,coords=coords).to_dataset(dim='variable') # clumsy but I am a xr noobie okay
        
        #########################################################################
        #########################################################################
        ########################################################### CONTINUE HERE
        
        # only keep one record every ::logperiod for faster processing of diffusion

        if keeplog is False:
            
            xvfc_we = xvfc_we.isel(time=-1) #only keep the last iteration

        else:
            # logperiod tells the frequency at which to keep a trace of the vfc
            # in future version, could be a slice of df['date'] etc
            
            xvfc_we = xvfc_we.isel(time=slice(None,None,-1)).isel(time=slice(None,None,logperiod)).isel(time=slice(None,None,-1))


    # Add in physical parameters: density
    
    depthwe = xvfc_we['depth'].values
    Tmean = np.nanmean(Temp)
    depthsnow,rhod = fm.DensityHL_we(depthwe,rho,Tmean,accu)
    

    xvfc_we['rhoHL'] = xr.DataArray(rhod,coords={'depth':xvfc_we.coords['depth']})
    xvfc_we['depthwe'] = xr.DataArray(depthwe,coords={'depth':xvfc_we.coords['depth']}) # -> before switching to snow depth, store we depth inside dataset

    
    # INTRODUCE VFC IN REAL SNOW DEPTH
    
    xvfc = copy.deepcopy(xvfc_we)
    
    xvfc.coords['depth'] = depthsnow

    xvfc = xvfc.interp({'depth':np.arange(0,np.max(depthsnow),res)},method='linear')

    sigma_cm = {}
    sigma_cm['d18O'],sigma_cm['dD'] = fm.Diffusionlength_OLD(xvfc['depth'].values,xvfc['rhoHL'].values,Tmean,650,accu);

    xvfc['sigma18'] = xr.DataArray(sigma_cm['d18O'],coords={'depth':xvfc.coords['depth']})
    xvfc['sigmaD'] = xr.DataArray(sigma_cm['dD'],coords={'depth':xvfc.coords['depth']})

    #return xvfc,xvfc_we

   # now we can diffuse
    specs = [] # columns to diffuse
    sigmas = []
    for speci,sigmai in [('d18O','sigma18'),('dD','sigmaD')]:
        for column in xvfc.var(): # scan variables and assign species one more time
            if speci in column:
                specs.append(column)
                sigmas.append(sigmai)
    
    sigmaarray = np.stack([xvfc[sigmai] for sigmai in sigmas],axis=1)

    #return xvfc,xvfc_we

    if False:
        # Testing out ways to diffuse; each column one by one or all columns in parralel
        # odly it seems that all columns in parralel is slower

        # put depth first for the loop over depth in Diffuse_record
        # and put variable last so each species can pick up its own column in sigmaarray
    
        xvfc[[speci+'_diff' for speci in specs]] =  xr.DataArray(Diffuse_record2(xvfc[specs].to_dataarray().transpose('depth',...,'variable'),
                                                   sigmaarray/100/res,resolve='full'),
                                            coords = xvfc[specs].to_dataarray().transpose('depth',...,'variable').coords
                                                            ).transpose('variable',...,'depth').to_dataset(dim='variable')

    else:
        for i,speci in enumerate(specs):
            xvfc[speci+'_diff'] = xr.DataArray(fm.Diffuse_record2(xvfc[speci].T,sigmaarray[:,i]/100/res,resolve='full').T,
                                              coords = xvfc[speci].coords);
        
    #     if spec in storage_diffusion_cm.keys():
    #         sigma_storage_cm = (sigma_cm[spec]**(2) + storage_diffusion_cm[spec]**(2))**(1/2)   #d'après Dallmayre et al. (2024)
    #         vfc_snow_even[spec+'_diff2'] = fm.Diffuse_record_OLD(vfc_snow_even[spec],sigma_storage_cm/100,res)[0];
    #         vfc_snow_even['sigma'+spec+'_stored'] = sigma_storage_cm/100

    # compute dexc if it was part of the input
    
    if 'dD_diff' in list(xvfc.var()) and 'd18O_diff' in list(xvfc.var()):
        xvfc['dexc'] = xvfc['dD'] - 8 * xvfc['d18O']
        xvfc['dexc_diff'] = xvfc['dD_diff'] - 8 * xvfc['d18O_diff']
        xvfc['dexc_raw_diff'] = xvfc['dD_raw_diff'] - 8 * xvfc['d18O_raw_diff']

    return xvfc
