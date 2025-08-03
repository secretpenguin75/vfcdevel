import numpy as np
import nitime
import scipy
import scipy.stats
import copy

def PSD_SP(df,fs = None):
    
    # takes an input df and returns the mean PSD, PSD of mean and PSD's of columns
    
    # df is assumed to be already equaly sampled AND filled

    # P dataframe of powerspectra
    # Mean of power spectrum
    # S power spectrum of mean
    # Mn = M/n Mean power spectrum divided by number of arrays

    
    npits = len(df.T)
    
    ff,S = mtm_psd(df.mean(axis=1),fs)
    
    P = np.full((len(S),npits),np.nan)
    
    for i,column in enumerate(df.columns):
        P[:,i] = mtm_psd(df[column],fs)[1]
    
    return ff,S,P


def get_SNR(S,P):
    
    npits = P.shape[1]
    
    M = np.nanmean(P,axis=1)

    C = S-M/npits
    N = (M-S)
    
    SNR = np.maximum(C/N,np.full(C.shape,0)) # reject negative values
    
    return C,N,SNR

def mtm_psd(series,fs = None,detrend = True,NW = 2.5):
    # a simple wrapper for nitime multi-taper-method-psd

    dd = 365.25

    
    if fs is None:
        dt = np.diff(series.index)[0]
        if type(dt) == np.timedelta64:
            dt = dt/np.timedelta64(1,'D')
            dt /= dd # in units of per year
            fs = 1/dt
        else:
            fs = 1/np.diff(series.index)[0] # assume equaly sampled

        fs = abs(fs)
    
    signal = copy.deepcopy(series.to_numpy())
    
    # linear detrending
    if detrend:
        xs = np.array(range(len(signal)))
        a,b = scipy.stats.linregress(xs, signal, alternative='two-sided')[:2]
        trend = a*xs+b
        signal -= trend
    
    ff,psd = nitime.algorithms.multi_taper_psd(
    signal,NW = NW,adaptive=False, jackknife=False, Fs = fs, sides = 'onesided'
    )[:2]
        
    return ff,psd