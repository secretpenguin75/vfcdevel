import datetime as datetime
import pandas as pd
import numpy as np
import copy
import scipy

def read_species(Proxies):
    
    species = {}
    for key in Proxies.columns:
        for spec in ['d18O','dD','dexc']:
            if spec in key:
                species[key] = spec

    return species

def band_mask(n,dmin,dmax):
    dist = (np.add.outer(np.arange(n), -np.arange(n)))
    mask = np.logical_and(dist>=dmin,dist<dmax)
    return mask


##################################################
# stuff that has to do with averaging depth arrays

def core_sample_adrien(df_in,c_resolution,res = None):

    df = copy.deepcopy(df_in)
    samplingscheme = np.concatenate([np.arange(c[0],c[1]-c[2],c[2]) for c in c_resolution]) # convert c_resolution to sampling scheme = depth axis
    
    ind = np.digitize(df.index,samplingscheme,right=False)-1 # put depth array values into bins of the sampling scheme
    depth = np.unique(samplingscheme[ind])
    
    df['ind'] = ind
    
    out = df.groupby(df['ind']).mean()
    out = out.set_index(samplingscheme[out.index.to_numpy()])

    if not res is None:
        # lets you recover an even dataframe if this is what you prefer
        # if res is none, just returns the z-axis of the sampling
        out = df_interp(out,np.arange(min(out.index),max(out.index),res),kind='previous')
    
    return out

def core_sample_emma(df,c_resolution):
    
    #block average
    
    df = df.set_index (df.index + np.diff(df.index)[-1]/2)
    df_int = block_average_OLD(df,.01) # block average at 1cm resolution  
    
    df_int_list = []
    
    for i in range(len(c_resolution)):
        df_int_i = block_average_OLD(df[(df.index >= c_resolution[i][0]) & (df.index < c_resolution[i][1])],c_resolution[i][2])
        df_int_list.append(df_int_i)
    df_int = pd.concat(df_int_list)       
          
    return df_int

def array_interp(newindex,oldindex,array,kind='linear'):
    
    f = scipy.interpolate.interp1d(oldindex,array,kind=kind,fill_value='extrapolate',axis=0)
    
    array2 = f(newindex)

    return array2
    

def df_interp(df,newindex,kind='linear'):

    # Note that the point of passing via indices is that columns attribution
    # in DataFrame is slow, so doing it {number of columns} times is much slower
    # than doing it on an array with n columns

    df = df.sort_index()

    array = np.array(df)
    oldindex = df.index.to_numpy()
    
    array2 = array_interp(newindex,oldindex,array,kind)

    df2 = pd.DataFrame(array2,index = newindex,columns = df.columns)
    df2.index.name = df.index.name

    return df2

def block_average_TEST(df,block_scale,res = 1e-2):
    
    # combines the two steps of block averaging an array with non even block index
    # first interpolates to a subresolution with 'next' interpolation which applies to cumsumed precip input
    # then apply block averaging at block_scale resolution

    newindex = np.arange(0,np.max(df.index),block_scale*res)
    
    out = array_interp(newindex,df.index.to_numpy(),np.array(df),kind='next')
    out = block_average_e(out,int(1/res))
    out = pd.DataFrame(out,index = newindex[::int(1/res)],columns = df.columns)

    return out

def block_average_e(df,block_scale):

    # returns a block average of an array, pd.Series or DataFrame, regardless of the index
    # for integer value block scale, first interpolate on an even grid.
    
    array = np.array(df)

    #if array.shape[0]%cc != 0:
    #    raise Exception("block_scale must be an integer divisor of len(df)!") 

    # ...or add missing rows at the end to have integer multiple of block_scale
    # (in order to use numpy reshape)
    
    
    n = (block_scale-len(array)%block_scale)%block_scale

    extension = np.full([n]+list(array.shape[1:]),np.nan)
    
    array = np.concatenate([array,np.full([n]+list(array.shape[1:]),np.nan)],axis=0)
    
    cc = block_scale
    bb = array.shape[0]//cc

    out = array.reshape([bb,cc]+list(array.shape[1:]))

    out = np.mean(out,axis=1)

    if type(df) == type(pd.DataFrame()):
        out = pd.DataFrame(out,columns = df.columns,index= df.index[::cc])
    if type(df) == type(pd.Series()):
        out = pd.Series(out,index = df.index[::cc])
        
    return out


#def block_average_OLD2(df,res):
#    # a wrapper for core sample which a fast way to block average arrays
#    df1 = df_interp(df,np.arange(0,np.max(df.index),res),kind='next')
#    out = core_sample_adrien(df1,[[0,np.max(df1.index),res]])
#    
#    return out

def block_average_OLD(df,res):

    # The older version would not create a dataframe with a value at each step
    # it would only create blocks for values in the initial dataframe
    
    # With intermediate interpolation we ensure that the grid is regular for the block averaged dataframe
    
    # takes as input a dataframe with a depth (in meters) columns
    # and output resolution (in meters)
    # returns the block average of the dataframe as the given resolution

    #newindex = np.arange(min(df.index),max(df.index),res/100)
    newindex = np.arange(0,max(df.index),res/100)
    df1 = df_interp(df,newindex,kind='next')


    df2 = copy.deepcopy(df1)
    index = np.array(df2.index)

    indexname = df2.index.name
    
    #df2.index.name = None # to avoid error in case the df.index is already called "depth" and since index name will be overwritten anyways
    
    df2['xxxxx'] = ( index // res) * res

    out = df2.groupby('xxxxx').mean()

    out.index.name = indexname
    
    return out


def block_average_OLD3(df,res):
    
    # takes as input a dataframe with a depth (in meters) columns
    # and output resolution (in meters)
    # returns the block average of the dataframe as the given resolution


    df2 = copy.deepcopy(df)
    depth = np.array(df2.index)

    df2.index.name = None

    
    df2['depth'] = (depth // res) * res
    
    return df2.groupby('depth').mean()


def df_interp_notinuse(df,newindex):
    # There is a very weird bug with this version
    # gives me the error "cannot reindex on an axis with duplicate labels"
    # even though print(sum(compositeindex.duplicated())) gives 0...
    #????????

    compositeindex = pd.Index(list(df.index.to_numpy())+list(newindex))
    compositeindex = compositeindex.unique().sort_values() # I've had weird pandas bugs without this
    print(sum(compositeindex.duplicated()))
    return compositeindex
    df2 = df.reindex(compositeindex)
    df2 = df2.interpolate(method='values') # pandas interpolate will fill NaN values
    df2 = df2.loc[newindex] # only keep values of the new axis
    df2 = df2.drop_duplicates() # drop duplicates in case old and new indices overlap
    
    return df2

#def df_interp_OLD(df,newindex,kind='linear'):

#    df = df.sort_index()
#    
#    df2 = pd.DataFrame(index = newindex)
#    
#    for column in df.columns:
#        f = scipy.interpolate.interp1d(df.index.to_numpy(),df[column].to_numpy(),kind=kind,fill_value='extrapolate')
#        df2[column] = f(newindex)
#
#    return df2

#def df_interp_OLD(df,newindex):
#    
#    df2 = pd.DataFrame(index = newindex)
#    
#    for column in df.columns:
#        df2[column] = np.interp(newindex,df.index,df[column])
#
#    return df2



############################################
# everything that has to do with time arrays


def decimalyear_to_datetime(decimalyear):

    # from decimalyear (float) to datetime.datetime object
    
    year = int(decimalyear)
    
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_in_year = 366
    else:
        days_in_year = 365

    #cumulateddays = np.array([0]+list(np.cumsum(days_in_months)))


    #day = int(remainder*days_in_year)
    
    #ind = np.digitize(day,cumulateddays)
    #month = ind

    #day = (cumulateddays-day)[ind]

    #out = datetime.date(year,month,day)

    x = decimalyear

    out = datetime.datetime(int(x), 1, 1) + datetime.timedelta(days = (x % 1) * days_in_year)

    return out

def datetime_to_decimalyear(dtime):

    # works for either datetime.datetime or pandas.timestamp

    year = dtime.year
    
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_in_year = 366
    else:
        days_in_year = 365

    fraction  = (dtime - datetime.datetime(year,1,1,0,0))/datetime.timedelta(1)/days_in_year
    out = year + fraction
    
    return out

def str_to_decimalyear_OLD(string):
    
    year = int(string[0:4])
    month = int(string[5:7])
    day = int(string[8:10])

    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_in_year = 366
        days_in_months = np.array([31,29,31,30,31,30,31,31,30,31,30,31])
    else:
        days_in_year = 365
        days_in_months = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

    cumulateddays = np.array([0]+list(np.cumsum(days_in_months)))
    fraction = (cumulateddays[month-1]+(day-1))/days_in_year

    decimal_year = year+fraction
    
    return year+fraction



def ts_to_dt(ts_array):
    return [bobi.date() for bobi in ts_array]

def get_time_ref(ttype):

    if ttype == datetime.date:
        ref = datetime.date(2000,1,1)
    elif ttype == datetime.datetime:
        ref = datetime.datetime(2000,1,1)
    elif ttype == pd.Timestamp:
        ref = pd.Timestamp("2000-1-1")

    # DO NOT try to work with numpy datetime, things are too different
    #elif ttype == np.datetime64:
    #    ref = np.datetime64(30,'Y')

    return ref

def get_one_second(ttype):

    if ttype == datetime.date:
        out = datetime.timedelta(0,1) # one second
    elif ttype == datetime.datetime:
        out = datetime.timedelta(0,1) # one second
    elif ttype == pd.Timestamp:
        out = pd.Timedelta('1s') # one second

    return out

def to_seconds(time_object,ref = None):
    
    # function that returns the number of seconds since reference
    # and works with different types of datetimes
    
    if ref is None:
        ref = get_time_ref(type(time_object))

    one_second = get_one_second(type(time_object))
            
    return (time_object-ref)/one_second

def to_time(seconds,ref = None,optype = datetime.date):
    
    if ref is None:
        ref = get_time_ref(optype)
        
    # nb: output will inherit the type of ref, whatever the type of timedelta
    
    return ref + datetime.timedelta(0,seconds)

def float_time_interp(float_bins,float_list,time_list):

    # time list can be: a list of datetime, a list of date, a list of timestamps...
    
    float_list,time_list = zip(*sorted(zip(float_list,time_list))) # sort float x-array in ascending order
    
    seconds_list = list(map(to_seconds,time_list))

    out = np.interp(float_bins,float_list,seconds_list)

    
    out = list(map(lambda x:to_time(x,optype = type(time_list[0])),out))
    
    return out

def time_float_interp(time_bins,time_list,float_list):

    # time list can be: a list of datetime, a list of date, a list of timestamps...
    
    time_list,float_list = zip(*sorted(zip(time_list,float_list)))
    
    seconds_list = list(map(to_seconds,time_list))
    seconds_bins = list(map(to_seconds,time_bins))

    out = np.interp(seconds_bins,seconds_list,float_list)
    
    return out


# FUNCTIONS THAT HAVE TO DO WITH PRECIPITATION INPUT

def transfer_tp(TP,newindex):

    cumprecip = TP.cumsum()

    tp = np.diff(time_float_interp(newindex,cumprecip.index,cumprecip.values),prepend=np.nan)

    return tp


def precip_to_tp(time_index,precip):
    
    # instantaneous precipitation (kg/s or mm w.e/s) to total precipitation (mm w.e) for corresponding periods of the input time index
    
    cumprecip = accu_to_elev(precip) # hidden unit conversion, assumed precip is in kg per m^2 per _seconds_!
    
    tp = np.diff(time_float_interp(time_index,precip.index,cumprecip),prepend=np.nan)

    return tp



def weight_tp_M(time,signal,tp,window):

    # Mathieu's version of weight precip; takes a tail behind each datapoint to collect 1.5cm accumulation
    # Mathieu always works in total precipitation for the (preceding) period

    dep = window
    
    intprecip = tp # integrated precips

    out = np.full(len(signal),np.nan)
    for i in range(len(signal)):
        j=0
        while np.sum(intprecip.iloc[i-j:i])<dep and i-j>=0: # find the number of datapoints we need to add up
            j+=1

        
        A = np.nansum((signal*intprecip).iloc[i-j:i])
        B = np.nansum(intprecip.iloc[i-j:i])
        

        out[i] = A/B # weight selected datapoints by amount of precip

    return out


def weight_tp_A(time,signal,tp,window,shift=0,res=0.01):

    # assume precip is in units of per second
    
    # effect of precipitation intermitency
    # weights the signal by the quantity of precipitation
    # need to specify a window width for the quantity of precip to accumulate and average over

    # Adrien's version. here we see that we are really doing the same
    # as a VFC...
    
    dep = window

    elev = np.cumsum(tp)

    # preparing evenly spaced array for fast windowing
    
    #res = 0.01

    elev_e = np.arange(np.min(elev),np.max(elev),res) # to the hundredth of mm

    f2 = scipy.interpolate.interp1d(elev,signal,kind='next') # next; associate signal value to depth and all intermediate PRECEDING depths
    signal_e = f2(elev_e) # square shaped interpolation

    #plt.plot(depth_e,signal_e)

    # preparing for the rolling average
    window_e = int(np.ceil(window/res)) # window normalized in indices
    shift_e = int(np.ceil(shift/res)) # shift normalized in indices

    
    signal_e_ave = pd.Series(signal_e).rolling(window_e,min_periods=1).mean().shift(-shift_e) # min period 1 to avoid nan at the start of the time series
    #nb interestingly for us, rolling takes exactly the trailing window of observations: takes the average of the previous values over the range of _window_ mm w.e.
    #this is exactly the behaviour that we want in terms of accumulation (forward in time)
    
    signal_out = np.interp(elev,elev_e,signal_e_ave)
    
    return signal_out

def wtp_df(df,signal_column,tp_column,window,shift=0):
    
    out = pd.DataFrame(weight_tp_A(df.index,df[signal_column],df[tp_column].shift(shift),window)).set_index(df.index)
    
    return out

# alternative version that works with precipitaiton rates (divided by time delta)

def weight_precip_A(time,signal,precip,window,shift=0,res=0.01):

    #time, signal and precip are assumed to be the same shape
    # window in depth units
    # shift in depth units

    # assume precip is in units of per second
    
    # effect of precipitation intermitency
    # weights the signal by the quantity of precipitation
    # need to specify a window width for the quantity of precip to accumulate and average over

    # Adrien's version. here we see that we are really doing the same
    # as a VFC...
    
    dep = window
    
    dt = np.diff(time).astype('timedelta64[s]').astype(int) # convert to number of days, and we will assume precip is in units of per seconds
    dt = np.concatenate([[dt[0]],dt]) # restore first value, assume it is similar to second value (daily, monthly...)

    intprecip = precip*dt # integrated precips

    depth = np.cumsum(intprecip) # to meters

    # preparing evenly spaced array for fast windowing
    
    #res = 0.01

    depth_e = np.arange(np.min(depth),np.max(depth),res) # to the hundredth of mm

    f2 = scipy.interpolate.interp1d(depth,signal,kind='next') # for staircase square interpolation
    signal_e = f2(depth_e) # square interpolation

    # preparing for the rolling average
    window_e = int(np.ceil(window/res)) # window normalized in indices
    shift_e = int(np.ceil(shift/res)) # shift normalized in indices

    # signal will typically be given 
    # to get average value e.g 20mm under the surface we will shift up by 20mm after rolling -> negative sign in shift
    
    signal_e_ave = pd.Series(signal_e).rolling(window_e,min_periods=1).mean().shift(-shift_e) # min period 1 to avoid nan at the start of the time series
    
    signal_out = np.interp(depth,depth_e,signal_e_ave)
    
    return signal_out

def wp_df(df,signal_column,precip_column,window,shift=0):
    
    out = pd.DataFrame(weight_precip_A(df.index,df[signal_column],df[precip_column].shift(shift),window)).set_index(df.index)
    
    return out

# From elevation to accumulation and vice-versa
# some functions that I used in the trench accumulation study

def elev_to_accu(elev_df):

    # assume type of df index is datetime64[ns]


    # elevation = cumsum (accumulation x dT)
    # accumulation = d_elevation / dT
    
    # input: elevation df of length n
    # output: accumulation df of length n 
        
    dt_array = elev_df.index.to_series().diff().dt.days
        

    accu_df = (elev_df.diff(axis=0).T/dt_array.to_numpy()).T

    #accu_df = accu_df.reindex_like(elev_df) # otherwise datetime type is changed???
    
    return accu_df

def accu_to_elev(accu_df,elev_df_top = None):

    # assume accu is in units of per seconds!!!
    
    # input: accu df with k columns and time index of length n, x0 array of length k
    # output: cumsum(dx/dt) + x0 array of length n

    # meant to work with 1d and 2d arrays/dataframes;
    
    if elev_df_top is None:
        if len(accu_df.shape)>1:
            elev_df_top = np.full(accu_df.shape[1],0)
        else:
            elev_df_top = np.array([0])

    dt = np.diff(accu_df.index).astype('timedelta64[s]')/np.timedelta64(1,'s') # convert to number of days, and we will assume precip is in units of per seconds
    dt = np.concatenate([[np.nan],dt]) # dummy last value: traditionally, tp value of 01-01-2000 is from 01-01-2000 to 01-02-2000 -> last value not associated to time delta

    delev_array = (accu_df.mul(dt,axis='rows')).to_numpy() 
    
    # change of elevation = accumulation X dt
    # first value of accumulation is lost; we don't know what time delta it relates to

    # first value of delev_array (nan) is replaced by elev_df_top, which is zero by default, but can be changed to something else


    delev_df = pd.DataFrame(delev_array).set_index(accu_df.index)

    delev_df.shift() # shift by one
    
    delev_df.iloc[0] = elev_df_top

    elev_df = delev_df.cumsum()

    # just some manipulation for the output
    # if input is pd timeseries: return timeseries
    # if input is pd dataframe: return dataframe
    
    if type(accu_df) == pd.DataFrame:
        elev_df.columns = accu_df.columns # restore column names if input is Dataframe
    elif type(accu_df) == pd.Series:
        elev_df = elev_df[0] # extract first column if input was just a series :)
    

    #elev_df = elev_df.reindex_like(accu_df) # otherwise datetime type is changed???

    
    return elev_df