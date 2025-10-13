import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import copy


def xpcolormesh(xarray,dx=1,dy=1,vmin=None,vmax=None,cmap=None):
    
    a = copy.deepcopy(xarray)
    xlabel = list(a.coords)[0]
    ylabel = list(a.coords)[1]
    x = a.coords[xlabel].values
    y = a.coords[ylabel].values
    plt.pcolormesh(x[::dx],y[::dy],a[::dx,::dy].T,vmin=vmin,vmax=vmax,cmap=cmap)
    plt.gca().set_ylabel(ylabel)
    if 'depth' in ylabel:
        plt.gca().invert_yaxis()

def dflip0(array,axis=0):

    #hacky but straightforward
    #only works for evenly sampled array (depth has same step as elev)

    mask = np.isfinite(array.astype(float))
    out = np.full(array.shape,np.nan)
    
    out[np.flip(mask,axis=axis)] = array[mask] # crux
    out = np.flip(out,axis=axis)

    return out

def dflip(xrarray):
    # a wrapper for numpy dflip that manages dimensions names 'depth' and 'elev'

    xrin = xrarray

    dim = list(set(['depth','elev']) & set(xrin.coords))
    olddim = dim[0]
    newdim = list(set(['depth','elev']) - set(dim))[0]

    i = list(xrin.coords).index(olddim)
        
    #mask = np.isfinite(np.array(xrin).astype(float))
    #out = np.full(xrin.shape,np.nan)
    
    #out[np.flip(mask,axis=i)] = np.array(xrin)[mask] # crux
    #out = np.flip(out,axis=i)

    out = dflip0(np.array(xrin),i)
    
    xrout = xr.DataArray(out,coords = xrin.coords,dims=xrin.dims)

    xrout = xrout.rename({olddim:newdim})
    
    return xrout

def dflipds(xrdataset):
    
    # I couldnt figure out how to dflip the dataset so we just dflip each variable with a loop

    #dim = list(set(['depth','elev']) & set(xrdataset.coords)) # find out wether depth or elev are used in the dataset
    #olddim = dim[0]
    #newdim = list(set(['depth','elev']) - set(dim))[0]

    out = {}
    
    for key in xrdataset:
        out[key] = dflip(xrdataset[key])

    out = xr.Dataset(out)

    #out = out.drop_dims(olddim)

    return out


