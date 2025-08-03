import datetime as datetime
import pandas as pd
import numpy as np
import copy
import scipy

def band_mask(n,dmin,dmax):
    dist = (np.add.outer(np.arange(n), -np.arange(n)))
    mask = np.logical_and(dist>=dmin,dist<dmax)
    return mask


##################################################
# stuff that has to do with averaging depth arrays

def core_sample_adrien(df,c_resolution):

    samplingscheme = np.concatenate([np.arange(c[0],c[1]-c[2],c[2]) for c in c_resolution]) # generate sampling scheme fore core resolution
    
    ind = np.digitize(df.index,samplingscheme,right=False)-1 # put depth array values into bins of the sampling scheme
    depth = np.unique(samplingscheme[ind])
    df['ind'] = ind
    out = df.groupby(df['ind']).mean()
    out = out.set_index(samplingscheme[out.index.to_numpy()])
    
    return out

def core_sample_emma(df,c_resolution):
    
    #block average
    
    df = df.set_index (df.index + np.diff(df.index)[-1]/2)
    df_int = block_average(df,.01) # block average at 1cm resolution  
    
    df_int_list = []
    
    for i in range(len(c_resolution)):
        df_int_i = block_average(df[(df.index >= c_resolution[i][0]) & (df.index < c_resolution[i][1])],c_resolution[i][2])
        df_int_list.append(df_int_i)
    df_int = pd.concat(df_int_list)       
          
    return df_int


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

def block_average(df,res):

    # the older version would not create a dataframe with a value at each step
    # it would only create blocks for values in the initial dataframe
    # with intermediate interpolation we ensure that the grid is regular for the block averaged dataframe
    
    # takes as input a dataframe with a depth (in meters) columns
    # and output resolution (in meters)
    # returns the block average of the dataframe as the given resolution

    newindex = np.arange(min(df.index),max(df.index),res/100)
    df1 = df_interp(df,newindex,kind='next')


    df2 = copy.deepcopy(df1)
    depth = np.array(df2.index)

    df2.index.name = None

    
    df2.index.name = None # to avoid error in case the df.index is already called "depth" and since index name will be overwritten anyways
    
    df2['depth'] = (depth // res) * res
    
    return df2.groupby('depth').mean()


def block_average_OLD(df,res):
    
    # takes as input a dataframe with a depth (in meters) columns
    # and output resolution (in meters)
    # returns the block average of the dataframe as the given resolution


    df2 = copy.deepcopy(df)
    depth = np.array(df2.index)

    df2.index.name = None

    
    df2['depth'] = (depth // res) * res
    
    return df2.groupby('depth').mean()

def df_interp(df,newindex,kind='linear'):
    
    df2 = pd.DataFrame(index = newindex)
    
    for column in df.columns:
        f = scipy.interpolate.interp1d(df.index,df[column],kind=kind,fill_value='extrapolate')
        df2[column] = f(newindex)

    return df2

#def df_interp(df,newindex):
#    
#    df2 = pd.DataFrame(index = newindex)
#    
#    for column in df.columns:
#        df2[column] = np.interp(newindex,df.index,df[column])
#
#    return df2

############################################
# everything that has to do with time arrays


def decimalyear_to_str(decimal_year):
    
    year = int(decimal_year)
    remainder = decimal_year - year
    
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_in_year = 366
        days_in_months = np.array([31,29,31,30,31,30,31,31,30,31,30,31])

    else:
        days_in_year = 365
        days_in_months = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

    cumulateddays = np.array([0]+list(np.cumsum(days_in_months)))


    day = int(remainder*days_in_year)
    ind = np.argmax(cumulateddays-day>0)
    month = ind+1
    day = (cumulateddays-day)[ind]

    out = datetime.date(year,month,day)

    return out

def str_to_decimalyear(string):
    
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