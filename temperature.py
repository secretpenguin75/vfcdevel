import numpy as np

# contains the functions for calculating temperature profile at each step, with Crank-Nicholson scheme

def CN_RHS(T,alpha,a=None,b=None):

    out = np.full(len(T),np.nan)
    
    if not (a is None or b is None):
        out[0] = a
        out[-1] = b
        
    else:
        # if bc not specified just assumed Neumann BC
        out[0] = alpha*T[1] + (1-alpha) * T[0]
        out[-1] = alpha*T[-2] + (1-alpha) * T[-1]
        
    out[1:-1] = alpha/2*T[2:] + (1-alpha) * T[1:-1] + alpha/2 * T[:-2]
    return out

def CN_matrix(n,alpha,bc='Neumann'):
    
    if bc == 'fixed':
        out = np.diagflat([-alpha/2 for i in range(n-2)]+[0.], -1) +\
          np.diagflat([1.]+[1.+alpha for i in range(n-2)]+[1.]) +\
          np.diagflat([0.]+[-alpha/2 for i in range(n-2)], 1)

    elif bc == 'Neumann':
        out = np.diagflat([-alpha/2 for i in range(n-2)]+[alpha], -1) +\
          np.diagflat([1.+alpha for i in range(n)]) +\
          np.diagflat([0.]+[-alpha/2 for i in range(n-2)], 1)

    return out

def temperature_step(Tin,depth,tsurf,D,dt,cc = None):
    
    # Tin temperature profile over depth array at beginning of step
    # depth array
    # new surface temperature
    # snow diffusivity (Profile with rho dependance in the future??)
    # dt timestep
    # cc slice on which to compute the temperature profile
    # if dt is daily: cant do better than 25cm, use cc = [np.arange(0,2000,25)]
    # if dt is hourly: can reach 5cm, but we dont want to compute at 5cm res down to 20m, 
    # so use [np.arange(0,2000,25),np.arange(0,200,5)]

    Tout = np.full(depth.shape,np.nan)

    if cc is None:
        cc = [np.arange(len(depth))] # use the entire depth range
    
    for c in cc:
            
        dzc = np.diff(depth[c])[0]
        
        alpha = D*dt/dzc**2

        #CONTINUE HERE
        b = CN_RHS(Tin[c],alpha,tsurf,Tin[c[-1]])
        
        A = CN_matrix(len(c),alpha,bc='fixed')

        Tout[c] = np.linalg.solve(A,b)

    
    c4 = np.unique(np.sort(np.concatenate(cc)))
    
    Tout = np.interp(depth,depth[c4],Tout[c4])

    return Tout