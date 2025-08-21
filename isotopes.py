def temp_to_iso_DC(Temp,iso,fit='era5'):

    # input: Temperature in Kelvin

    #Definition of the isotopic composition - temperature relationship
    if fit=='Mathieu':
        alpha = 0.46; 
        beta = -32;
        
    if fit=='era5':
        alpha = 0.51
        beta=-29

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