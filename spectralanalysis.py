import numpy as np
import nitime
import scipy
import scipy.stats
import copy

import pywt

import matplotlib.pyplot as plt

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


# wavelets

def wavelets_adrien(df):
    
    # Extraire l axe spatial (x) et le signal
    #x_orig = df.index.values# / 1000
    #signal_orig = df.values
    
    # Interpolation sur un axe regulier
    #x_uniform = np.linspace(x_orig.min(), x_orig.max(), len(x_orig))
    #x_uniform = np.arange(x_orig.min(), x_orig.max(), 0.01)

    #interpolator = scipy.interpolate.interp1d(x_orig, signal_orig, kind='linear')
    #signal_uniform = interpolator(x_uniform) - np.mean(interpolator(x_uniform))

    # we just assume df is uniform
    x_uniform = df.index.values
    signal_uniform = df.values

    # dont forget to remove the mean!! important for wavelet analysis
    signal_uniform = signal_uniform-np.nanmean(signal_uniform)

    
    dx_uniform = x_uniform[1] - x_uniform[0]

    
    # Analyse en ondelettes continues (CWT)
    # fs = 1/dz sampling frequency
    # scale = 1 -> frequency is fs / 1
    # scale = 2 -> frequency is fs / 2
    # scale = max(depth)/fs -> frequency is 1/max(depth) (maximum freq in fourier)
    # number of scales per octave = 32: scales = 2,2**(1+1/32), 2**(1+2/32),...,2**2,... etc...
    

    max_period = x_uniform.max()
    
    log2maxscale = -np.log(dx_uniform/max_period)/np.log(2)
    octa = 16 # number of scales per octave
    scales = 2**(np.arange(1,log2maxscale,1/octa))
    
    #scales = np.arange(1,254) # in Amaelle and Emma's version
    
    wavelet = 'cmor1.5-1.0'  # Morlet complexe

    # apply pywt function
    coefficients, frequencies = pywt.cwt(signal_uniform, scales, wavelet, sampling_period=dx_uniform)
    
    # Analyse en FFT
    #N = len(signal_uniform)
    #fft_amplitudes = 2.0 / N * np.abs(scipy.fft.fft(signal_uniform)[:N // 2])
    #fft_freqs = scipy.fft.fftfreq(N, d=dx_uniform)[:N // 2]# Analyse en FFT
    
    # Spectres

    # J'aime bien séparer les fonctions qui calculent les spectres et les fonctions qui plotent, donc je m'arrête ici
    return frequencies,coefficients
    
    #plt.figure(figsize=(10, 7))
    
    #plt.subplot(111)
    #print('frequencies',frequencies)
    
    #plt.imshow(np.abs(coefficients), extent=[x_uniform[0], x_uniform[-1], frequencies[-1], frequencies[0]], aspect='auto', cmap='viridis')
    # warning: extent is only valid for regular grid
    # but the output of the pywt function is not a even scale of frequencies

    # instead we should do
    #plt.imshow(np.abs(coefficients), aspect='auto', cmap='viridis',vmin = 0,vmax = 5)
    #plt.gca().set_xticks(range(len(x_uniform))[::1000],x_uniform[::1000])
    #plt.gca().set_yticks(range(len(frequencies))[::20],frequencies[::20])
    #plt.gca().set_ylim(100,0.41)
    # but let's maybe switch to pcolormesh on a log scale

    #plt.pcolormesh(x_uniform,frequencies,np.abs(coefficients),vmin=0,vmax=5,rasterized=True)
    #plt.gca().set_ylim(0.41,100)
    #plt.gca().set_ylim(1/x_orig.max(),1/dx_uniform/2)
    #plt.gca().set_yscale('log')


    
    #plt.colorbar(label='Amplitude')
    #plt.ylabel('Spatial frequency [$m^{-1}$]')
    #plt.xlabel('Position [$m$]')
    
    #plt.subplot(212)
    #plt.plot(fft_freqs, fft_amplitudes, color='k')
    #plt.xlim(0, np.max(frequencies))  # meme echelle que la CWT
    #plt.xlim(0, 10)
    #plt.xlabel('Spatial frequency [$m^{-1}$]')
    #plt.ylabel('Amplitude')
    #plt.grid(True)
    #plt.subplots_adjust(hspace=0.3)

def wavelets_fft_spectra(df, name,newversion=False):

    oldversion = not newversion

    #oldversion = True
    
    # Extraire l axe spatial (x) et le signal
    x_orig = df.index.values# / 1000
    signal_orig = df['d18O'].values
    
    # Interpolation sur un axe regulier
    if oldversion:
        x_uniform = np.linspace(x_orig.min(), x_orig.max(), len(x_orig)) # in Amaelle and Emma's version
    else:
        x_uniform = np.arange(x_orig.min(), x_orig.max(), 0.01) # with set resolution

    interpolator = scipy.interpolate.interp1d(x_orig, signal_orig, kind='linear')
    signal_uniform = interpolator(x_uniform) - np.mean(interpolator(x_uniform))
    
    dx_uniform = x_uniform[1] - x_uniform[0]
    
    # Analyse en ondelettes continues (Wavelet)
    scales = np.arange(1, 254) # arbitrary set of scales?
    
    wavelet = 'cmor1.5-1.0'  # Morlet complexe
    coefficients, frequencies = pywt.cwt(signal_uniform, scales, wavelet, sampling_period=dx_uniform)
    
    # Analyse en FFT
    N = len(signal_uniform)
    fft_amplitudes = 2.0 / N * np.abs(scipy.fft.fft(signal_uniform)[:N // 2])
    fft_freqs = scipy.fft.fftfreq(N, d=dx_uniform)[:N // 2]
    
    # Spectres
    
    plt.figure(figsize=(10, 7))
    
    plt.subplot(211)
    
    if oldversion:
        plt.imshow(np.abs(coefficients), extent=[x_uniform[0], x_uniform[-1], frequencies[-1], frequencies[0]], aspect='auto', cmap='viridis')

    else:
        # warning: extent is only valid for regular grid
        # but the output of the pywt function is not a even scale of frequencies
    
        # instead we should do
        plt.imshow(np.abs(coefficients), aspect='auto', cmap='viridis')#,vmin = 0,vmax = 5)
        plt.gca().set_xticks(range(len(x_uniform))[::1000],x_uniform[::1000])
        plt.gca().set_yticks(range(len(frequencies))[::20],frequencies[::20])
        # but let's maybe switch to pcolormesh on a log scale

        #plt.pcolormesh(x_uniform,frequencies,np.abs(coefficients),vmin=0,vmax=5,rasterized=True)
        #plt.gca().set_ylim(1/x_orig.max(),1/dx_uniform/2)
        #plt.gca().set_yscale('log')

    
    plt.colorbar(label='Amplitude')
    #plt.ylim(0,100)
    plt.ylabel('Spatial frequency [$m^{-1}$]')
    plt.xlabel('Position [$m$]')
    plt.title(name)
    
    plt.subplot(212)
    plt.plot(fft_freqs, fft_amplitudes, color='k')
    if oldversion:
        plt.xlim(0, np.max(frequencies))  # meme echelle que la CWT
        plt.xlim(0, 10)
    else:
        pass
        
    plt.xlabel('Spatial frequency [$m^{-1}$]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.subplots_adjust(hspace=0.3)