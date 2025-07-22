import numpy as np

def logsmooth(ppx,f,dflog,remove_first = 1):
        
    dof = 1
    f_out = f[remove_first:]
    ppx_out,dof_out = smoothlogCE(ppx[remove_first:],f_out,dflog,dof);
    dof_out = dof_out
    ppx_out = ppx_out

    return ppx_out,f_out,dof_out


## SUBFUNCTIONS

def weights(x, sigma):
    #print('sigma',sigma)
    w = 1/np.sqrt(2 * np.pi * sigma**2) * np.exp(-x**2/(2 *  sigma**2))
    return w

def fweights(ftarget,f,dflog):
    sigma = ftarget*(np.exp(dflog)-np.exp(-dflog))
    fw = weights(f-ftarget,sigma)
    return fw

def smoothlog(x,f,dflog):
    x_out = np.full(np.nan,len(f));
    for i in range(len(f)):
        w = fweights(f[i],f,dflog);
        x_out[i] = np.sum(x*(w/np.sum(w)));
    return x_out

def smoothlogCE(x,f,dflog,dof):
    x_out = np.full(len(f),np.nan);
    dof_out = np.full(len(f),np.nan);
    for j in range(len(f)):
        w =  fweights(f[j],f,dflog);
        DistanceSlowEnd = j;
        DistanceFastEnd = len(f)-j-1;
   
        if (j+DistanceSlowEnd + 2)<=len(f):
            w[j+DistanceSlowEnd+1:] = 0;
        
        if j-DistanceFastEnd >= 1:
            w[:j-DistanceFastEnd-1] = 0;
            
        w = w/f;
        w = w/np.sum(w);
        x_out[j] = np.sum(x*w);
        dof_out[j] = np.sum(w*dof)/np.sum(w**2);       

    return x_out,dof_out
