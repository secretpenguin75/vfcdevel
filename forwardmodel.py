#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import scipy



# In[2]:

def DensityHL_we(depthwe,rho,Tmean,accu):

    # Just a wrapper for fm.densityHL that takes as input depth in mmwe
    # There are probably more efficient ways to proceed...

    depthsnow_uncompacted = depthwe*1000/rho # water depth is converted to uncompacted snow depth with the convertion factor 1000/rho
    # this uncompacted snow depth will overshoot the actual depth
    # this gives us an upperbound on the range over which to compute HerronLangway density profile
    
    depthwe_HL,rhod =  DensityHL(depthsnow_uncompacted,rho,Tmean,accu); #on calcule HL et la profondeur en we
    
    depthsnow = np.interp(depthwe,depthwe_HL,depthsnow_uncompacted)
    rhod = np.interp(depthwe,depthwe_HL,rhod)

    return depthsnow,rhod

    

def DensityHL(depth,rho_surface,T,bdot):
    
    #Constants
    kRhoIce = 920      #Density of ice [kg/m^3]
    kRhoW = 1000       #Density of water [kg/m^3]
    kR = 8.314478      #Gas constant [J/(K * mol)]

    
    #Critical density of point between settling and creep-dominated stages
    kRhoC = 550

    #Herron-Langway Arrhenius rate constants
    k0 = 11 * np.exp(-10160 / (kR * T))
    k1 = 575 * np.exp(-21400 / (kR * T))

    #Rate constants for time-dependent densification
    #(original Eq. (4) in Herron and Langway et al. (1980))
    
    A = bdot / kRhoW
    c0 = k0 * A
    c1 = k1 * np.sqrt(A)

    #Rate constants for depth-dependent steady-state densification
    #(from converting the full time derivative to a depth derivative
    #neglecting the partial time derivative to get steady-state solution)
    
    d0 = c0 / bdot
    d1 = c1 / bdot

    facr0 = rho_surface / (kRhoIce - rho_surface)
    facrc = kRhoC / (kRhoIce - kRhoC)
    
    #Critical depth at which density reaches kRhoC
    
    zc = np.log(facrc / facr0) / (kRhoIce * d0)
    index_upper = np.argwhere(depth <= zc)
    index_lower = np.argwhere(depth > zc)
    
    #Steady-state density profile
    
    q = np.full(len(depth),np.nan);
    q[index_upper] = facr0 * np.exp(d0 * kRhoIce * depth[index_upper]);
    q[index_lower] = facrc * np.exp(d1 * kRhoIce* (depth[index_lower] - zc));
    
    rho_prof = kRhoIce * (q / (1+q)) ;   

    #Time when critical depth is reached
    tmp = (kRhoIce - rho_surface) / (kRhoIce - kRhoC);
    tc = np.log(tmp) / c0;
    
    
    # Steady-state time - water-equivalent depth relation
    t = np.full(len(depth),np.nan) 
    
    tmp = (kRhoIce - rho_surface) / (kRhoIce - rho_prof[index_upper]) 
    t[index_upper] = np.log(tmp) / c0 
    
    tmp = (kRhoIce - kRhoC) / (kRhoIce - rho_prof[index_lower]) 
    t[index_lower] = np.log(tmp) / c1 + tc 

    depthwe = A*t 
    
    return depthwe, rho_prof


# an attempt to rewrite the function following more closely the derivation in HL1980
# not working!! work in progress
def DensityHL_NOTINUSE(depth,rho_surface,T,bdot):
    
    #Constants
    
    kRhoIce = 917 #920      #Density of ice [kg/m^3]
    kRhoW = 1000       #Density of water [kg/m^3]
    kR = 8.314478      #Gas constant [J/(K * mol)]

    
    #Critical density of point between settling and creep-dominated stages
    kRhoC = 550

    Rho0 = rho_surface

    #Herron-Langway Arrhenius rate constants
    k0 = 11 * np.exp(-10160 / (kR * T)) # eq. 6a in HL1980
    k1 = 575 * np.exp(-21400 / (kR * T)) # eq. 6b in HL1980


    facrc = kRhoC / (kRhoIce - kRhoC)
    
    facr0 = Rho0 / (kRhoIce - Rho0)

    
    #Critical depth at which density reaches kRhoC = 550 (h0.55 in HL1980)
    
    zc = 1 / (kRhoIce / kRhoW * k0) * ( np.log(facrc) - np.log(facr0) ) 
                              
    index_upper = np.argwhere(depth <= zc)
    index_lower = np.argwhere(depth >  zc)

    #Rate constants for depth-dependent steady-state densification
    #(from converting the full time derivative to a depth derivative
    #neglecting the partial time derivative to get steady-state solution)
    
    #d0 = c0 / bdot
    #d1 = c1 / bdot

    #d0 = k0 / kRhoW
    #d1 = K1 / kRhoW
    
    
    #Steady-state density profile
    
    q = np.full(len(depth),np.nan);
    
    q[index_upper] = facr0 * np.exp(kRhoIce / kRhoW * k0 * depth[index_upper]);
    
    q[index_lower] = facrc * np.exp(kRhoIce / kRhoW * k1 * (depth[index_lower] - zc));
    
    rho_prof = kRhoIce * (q/ (1+q)) ;  # equation 7 and 10 in HL1980 with q = Z0 or Z1

    
    #Rate constants for time-dependent densification
    #(original Eq. (4) in Herron and Langway et al. (1980))
    
    A = bdot / kRhoW
    c0 = k0 * A
    c1 = k1 * np.sqrt(A)
    

    #Time when critical depth is reached
    
    tmp = (kRhoIce - Rho0) / (kRhoIce - kRhoC);
    tc = np.log(tmp) / c0;
    
    
    # Steady-state time - water-equivalent depth relation
    
    t = np.full(len(depth),np.nan) 
    
    tmp = (kRhoIce - rho_surface) / (kRhoIce - rho_prof[index_upper]) 
    t[index_upper] = np.log(tmp) / c0 
    
    tmp = (kRhoIce - kRhoC) / (kRhoIce - rho_prof[index_lower]) 
    t[index_lower] = np.log(tmp) / c1 + tc 

    depthwe = A*t 
    
    return depthwe, rho_prof


def kdiff(model,iso,n):
    if iso =='18O':
        return 1-(Diff_ratio_air(model,iso))**n
    if iso =='D':
        return 1-(Diff_ratio_air(model,iso))**n


def Diff_ratio_air(model,iso):
    # returns Di/D for some isotopic species
    
    if model =='Merlivat': #From Merlivat 1978
        if iso =='18O':
            diff_ratio = 0.9755
        if iso =='D':
            diff_ratio = 0.9723

    if model =='Cappa': #as used in Cappa 2003
        
        if iso =='18O':
            diff_ratio = 0.969
        if iso =='D':
            diff_ratio = 0.984

    return diff_ratio


def Frac_eq(model, iso ,T):
# Calculate the equilibrium fractionation coefficients
# Type is either "Maj" for Majoube et al, 1971, or "El" for Elehoj et al, 2011

    if model == 'Maj':
        if iso == '18O':
            alpha = np.exp(11.839/(T+273.15) -28.224*10**(-3))        #Coefficient de fractionnement à l'équilibre de O18 de Majoube 1971
        if iso == 'D':
            alpha = np.exp(16289/(T+273.15)**2 - 9.45*10**(-2))  
            #Coefficient de fractionnement à l'équilibre de D de Merlivat et Nief 1967 repris par Majoube 1971

    if model == 'El':
        if iso == '18O':
            alpha = np.exp(0.0831 - 49.192/(T+273.15)+8312.5/(T+273.15)**2)           #Coefficient de fractionnement à l'équilibre de O18 de Ellehoj 2013
        if iso == 'D':
            alpha = np.exp(0.2133 - 203.10/(T+273.15) + 48888/(T+273.15)**2)
            #Coefficient de fractionnement à l'équilibre de D de Ellehoj 2013

    return alpha


def GoffGratch(TA,phase):
##Calcul de Pression de vapeur saturante (Pa) par rapport au solide,  d'apres la relation de Goff and Gratch. 
# Pour plus de détail voir RelativeHum.m

#TA temperature de l'air en K,
#type = 'ice' ou 'liq'
    
    if phase == 'ice':
        
        PsatO=6.1173; #en hPa,  Pression de vapeur saturante par rapport à la glace à To=273.15 K
        logPsat= -9.09718*(273.15/TA-1) - 3.56654*np.log10(273.15/TA) + 0.876793*(1-TA/273.15) +np.log10(PsatO);
        
    if phase == 'liq':
        
        Psat0=1013.25; #en hPa 'steam-point' pressure at 1atm, T=373.15K
        logPsat= -7.90298*(373.16/TA-1) + 5.02808*np.log10(373.16/TA) - 1.3816*10**(-7)*(10**(11.344*(1-TA/373.16))-1)
        + 8.1328*10**(-3)*(10**(-3.49149*(373.16/TA-1))-1) + np.log10(Psat0);
                
    Psat=(10**logPsat)*100;

    return Psat

def Diffusivity(rho, T, P,iso):
    
    #DIFFUSIVITY Calculate the diffusivity in polar firn

    #Set physical constants
    kR = 8.314478;              # Gas constant (J/(K * mol))
    kM = 18.02e-3;              # molar weight of H2O molecule (kg/mol)
    kP0 = 1013.25;              # standard atmospheric pressure (mbar)
    kRhoIce = 920;              # density of ice (kg/m3)
    
    #Saturation vapour pressure over ice (Pa)
    psat = GoffGratch(T,'ice');
    
    #Tortuosity constant
    b = 1.3;

    #Calculate tortuosity

    if (rho<= kRhoIce/np.sqrt(b)):
        invtau = 1-b*(rho/kRhoIce)**2
    else:
        invtau = 0
    
    alpha = Frac_eq('Maj',iso,T);    #'Maj' for Majoube and Merlivat formulaes # 'El' for Ellehoj estimations

    #Water vapour diffusivity in air (m^2 s^-1)
    # (Hall and Pruppacher, 1976)
    
    Da = 2.11e-5*(T/273.15)**(1.94) * (kP0/P);      # Problem of pressure being the atmospheric pressure (? comment by Mathieu) 

    Dai = Da*Diff_ratio_air('Merlivat',iso); # Here is were we use D^i/D. These value here are from Merlivat1978
    
    #Firn isotope diffusivity in cm^2 s^-1 (Johnsen et al. 2000)
    
    Di = (kM * psat * invtau * Dai *(1/rho - 1/kRhoIce))/(kR * T * alpha)*1e4;
  
    return Di


def Diffusivity_OLD(rho, T, P):
    # OLD version; we cleaned things up a bit and re-introduced function on floats instead of arrays
    #DIFFUSIVITY Calculate the diffusivity in polar firn

    #Set physical constants
    kR = 8.314478;              # Gas constant (J/(K * mol))
    kM = 18.02e-3;              # molar weight of H2O molecule (kg/mol)
    kP0 = 1013.25;              # standard atmospheric pressure (mbar)
    kRhoIce = 920;              # density of ice (kg/m3)
    
    #Saturation vapour pressure over ice (Pa)
    psat = GoffGratch(T,'ice');
    
    #Tortuosity constant
    b = 1.3;
    
    alpha18 = Frac_eq('Maj','18O',T);    #'Maj' for Majoube and Merlivat formulaes
    alphaD = Frac_eq('Maj','D',T);      # 'El' for Ellehoj estimations

    #Water vapour diffusivity in air (m^2 s^-1)
    # (Hall and Pruppacher, 1976)
    
    Da = 2.11e-5*(T/273.15)**(1.94) * (kP0/P);      # Problem of pressure being the atmospheric pressure (? comment by Mathieu) 

    
    Da18 = Da*Diff_ratio_air('Merlivat','18O'); # Here is were we use D^i/D. These value here are from Merlivat1978
    DaD  = Da*Diff_ratio_air('Merlivat','D');
    
    #Calculate tortuosity

    invtau = np.zeros(len(rho))
    for i in range(len(rho)):
        if (rho[i]<= kRhoIce/np.sqrt(b)):
            invtau[i] = 1-b*(rho[i]/kRhoIce)**2
    
    #Firn isotope diffusivity in cm^2 s^-1 (Johnsen et al. 2000)
    
    D18 = (kM * psat * invtau * Da18 *(1/rho - 1/kRhoIce))/(kR * T * alpha18)*1e4;
    DD = (kM * psat * invtau * DaD *(1/rho - 1/kRhoIce))/(kR * T * alphaD)*1e4;
  
    return D18, DD


# In[7]:

def Diffusionlength_OLD(depth,rho, T, P, bdot ):

    #bdot: accumulation rate in mm water equivalent


    #Calculation of the diffusivity

    # !!! here I simplified while waiting to have full temperature profiles
    #if (len(T) != len(depth)):
    #print('CAUTION: T and depth have different lengths, T has been modified')
    #    T = T[0]*np.ones(len(depth));
    
    T = T*np.ones(len(depth))
      
    D18,DD = Diffusivity_OLD(rho,T,P) # units of cm^2 per second   


    if (len(rho) != len(depth)):
        # If rho and depth have different length
        # 100 is the closeoff depth 
        
        rho = np.arange(rho[0],rho[0]+(krhoW-rho[0]*depth[-1]/100,(krhoW-rho[0])*depth[-1]/100/(len(depth)-1)))   
        
        print('CAUTION: rho and depth have different lengths, rho has been modified');
    
    # Depth increments in (m), extended with the mean depth increment
    
    dz = np.append(np.diff(depth), np.mean(np.diff(depth))) ;   

    krhoW = 997; # kg.m^-3 Density of water 

    time_d = np.cumsum(dz/(bdot/krhoW)*(rho/krhoW));   # Time scale accounting for densification
    ts = time_d*365.25*24*3600;                         # Convert from years to seconds

    drho = np.diff(rho);                   # Density gradients
    dtdrho = np.diff(ts) / np.diff(rho);    # Temporal density differential
    drho = np.append(drho,drho[-1]);          # Fill unknown gradients at final depths
    dtdrho = np.append(dtdrho,dtdrho[-1]);


    # Integrate diffusivity along the density gradient to obtain the diffusion
    # length (cm)
    
    var18 = 2*rho**2.*dtdrho*D18;
    varD = 2*rho**2.*dtdrho*DD;
    
    var18 = np.cumsum(var18 * drho);
    varD = np.cumsum(varD * drho);
    
    sigma18 = np.sqrt(rho**(-2)*var18);
    sigmaD = np.sqrt(rho**(-2)*varD);


    return sigma18,sigmaD




def Diffusionlength(depth,rho, T, P, bdot ):


    #Calculation of the diffusivity

    # !!! here I simplified while waiting to have full temperature profiles
    #if (len(T) != len(depth)):
    #    print('CAUTION: T and depth have different lengths, T has been modified')
    #    T = T[0]*np.ones(len(depth));
    
    #T = T*np.ones(len(depth))
      
    #D18,DD = Diffusivity(rho,T,P) # units of cm^2 per second   


    #if (len(rho) != len(depth)):
    #    # If rho and depth have different length
    #    # 100 is the closeoff depth 
    #    
    #    rho = np.arange(rho[0],rho[0]+(krhoW-rho[0]*depth[-1]/100,(krhoW-rho[0])*depth[-1]/100/(len(depth)-1)))   
    #    
    #    print('CAUTION: rho and depth have different lengths, rho has been modified');
    
    # Depth increments in (m), extended with the mean depth increment
    
    dz = np.append(np.diff(depth), np.mean(np.diff(depth))) ;   

    krhoW = 997; # kg.m^-3 Density of water 

    time_d = np.cumsum(dz/(bdot/krhoW)*(rho/krhoW));   # Time scale accounting for densification
    ts = time_d*365.25*24*3600;                         # Convert from years to seconds

    drho = np.diff(rho);                   # Density gradients
    dtdrho = np.diff(ts) / np.diff(rho);    # Temporal density differential
    drho = np.append(drho,drho[-1]);          # Fill unknown gradients at final depths
    dtdrho = np.append(dtdrho,dtdrho[-1]);


    # Integrate diffusivity along the density gradient to obtain the diffusion
    # length (cm)
    
    var18 = 2*rho**2.*dtdrho*D18;
    varD = 2*rho**2.*dtdrho*DD;
    
    var18 = np.cumsum(var18 * drho);
    varD = np.cumsum(varD * drho);
    
    sigma18 = np.sqrt(rho**(-2)*var18);
    sigmaD = np.sqrt(rho**(-2)*varD);


    return sigma18,sigmaD


# In[10]:


def Diffuse_record_OLD( dX, sigma, res ):

    n = len(dX);

    #Scale the diffusion length according to the resolution of the record. 
    
    sigma = sigma/res;

    #Extend the record to avoid NA's from the diffuse at the end of the
    #diffused record
    
    dX = np.append(dX , np.full(int(10*np.floor(max(sigma))), np.mean(dX)) );

    dX_diff = np.full(n,np.nan);

    for i in range(n):
        imax = np.ceil(5 * sigma[i]);
        ran = np.array(np.arange((i-imax),(i+imax),1),dtype=int);
        ran = ran[np.argwhere(np.logical_and(ran>0,ran<=n))];
        relran = i-ran;
        kernel = np.exp(-(relran)**2./(2.*sigma[i]**2));
        kernel = kernel/np.sum(kernel);
        rec = dX[ran];
        dX_diff[i] = np.sum(rec*kernel);

    F = np.fft.fft(dX_diff);
    I = np.fft.fft(dX[:n]); 
    HF = I - F; 

    return dX_diff,HF
    
def Diffuse_record(dX, sigma_e):
    
    # a gaussian running window that works with variable lengths :-)

    # sigma_e should be given in index size

    n = len(dX);

    #Scale the diffusion length according to the resolution of the record. 
    #sigma = sigma/res;
    sigma = sigma_e
    
    dX_diff = np.full(n,np.nan);

    # this is just a running window with varying kernel size
    # -> could be reimplemented with matrix operations
    for i in range(n):
        imax = np.ceil(5 * sigma[i]);
        ran = np.array(np.arange((i-imax),(i+imax+1),1),dtype=int);
        ran = ran[np.argwhere(np.logical_and(ran>=0,ran<n))];
        relran = i-ran;
        kernel = np.exp(-(relran)**2./(2.*sigma[i]**2));
        kernel = kernel/np.sum(kernel);
        rec = dX[ran];
        dX_diff[i] = np.nansum(rec*kernel); #<--- j'ai changé np.sum en np.nansum ici

    return dX_diff



# time to change everything to multi-D
def Diffuse_record2(dX, sigma_e,resolve='full'):
    
    # a gaussian running window that works with variable lengths :-)

    # sigma_e should be given in index size

    n = len(dX); # n is the first index, assumed to be depth (requires to transpose dX beforehand)
    

    #Scale the diffusion length according to the resolution of the record. 
    #sigma = sigma/res;
    sigma = sigma_e

    #dX_diff = copy.deepcopy(dX);

    if resolve == 'full':
        ind = range(n)
        
    elif resolve == 'coarse':
        d = 10
        ind = list(range(0,n,d))+[n-1]
        
    elif resolve == 'sigma':
        j = 0
        ind = []
        while j < n:
            ind.append(j)
            j+= int(np.ceil(sigma[j]))
        if n-1 not in ind: ind.append(n-1)

    dX_diff = np.full((len(ind),*dX.shape[1:]),np.nan)

    # this is just a running window with varying kernel size
    # -> could be reimplemented with matrix operations

    timebar = tqdm(list(enumerate(ind)),'Processing diffusion loop',colour='green')

    ran0 = np.arange(len(dX)).repeat(np.prod(dX.shape[1:])).reshape(dX.shape)
    ran0 = ran0.astype(float) # so we can set some entries to np.nan
    
    for j,i in timebar:
        
        imax = np.ceil(5 * sigma[i]);
        
        ran = ran0.copy()

        relran = ran-i # relative range

        relran[relran<-imax] = np.nan
        relran[relran>+imax] = np.nan
        relran[np.isnan(dX)] = np.nan # to avoid weighting nan values
        kernel = np.exp(-(relran)**2./(2.*sigma[i]**2));
        kernel = kernel/np.nansum(kernel,axis=0);
        #rec = dX[ran].squeeze();
        dX_diff[j] = np.nansum((kernel*dX),axis=0);
        
    f = scipy.interpolate.interp1d(ind,dX_diff.T,bounds_error=False)
    dX_diff = f(np.arange(n)).T
        
    dX_diff[np.isnan(dX)] = np.nan # restore nan values

    return dX_diff

def diffuse_isotopes(iso,depth,signal):
    
    # wrapping function for all the above
    
    
    Tsite = -53 + 273.15 #(mean temperature at Dome C)
    accu = 27 # accumulation at dome C (kg/m^2/year)
    rhofirn = 350 #density of firn
    Psite = 650 # surface pressure
    
    depthwe,rhod = DensityHL(depth,rhofirn,Tsite,accu)
    sigma18,sigmaD = Diffusionlength(depth,rhod,Tsite,Psite,accu)
    res = (depth[1]-depth[0])*1e2 #assume constant depth, resolution (in cm)
    
    if iso == 'd18O': 
        out = Diffuse_record(signal,sigma18,res)[0];
    if iso == 'dexc': 
        out = Diffuse_record(signal,sigmaD,res)[0];
        
    return out



