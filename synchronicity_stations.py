# Santiago Duarte
# Febrero 2021
# https://github.com/fsduarte/chaotic-downscaling 

import datetime as dt # Work with datetime format
import math # Mathematic operations
import netCDF4 as nc # Work with NetCDF format files
import nolds # Nonlinear measures for dynamical systems
import numpy as np # Array processing for numbers, strings, records, and objects
import os # use OS functions
import pandas as pd # Work with Pandas DataFrames
import sklearn.metrics # Machine Learning in Python
import sklearn.neighbors # Machine Learning in Python
from operator import sub # Standard operators as functions
from toolz import curry # List processing tools and functional utilities
from numba import jit # Use GPU
import warnings # Ignore not important warnings
import time # Compare time in the simulation
from tqdm import tqdm # Progress Bar 
warnings.filterwarnings("ignore")

@jit 
def dynamics_analysis(x):   
    # Chaos analyis of the time series
    # Use Eckmann et al algorithm for lyapunov exponents and FNN for embedding dimension
    # Returns the time delay, the embedding dimension and the lyapunov spectrum
    x=x.flatten()
    lag=delay(x)
    mmax=2*int(np.floor(2*math.log10(len(x))))+1 
    fnn=global_false_nearest_neighbors(x, lag, min_dims=1, max_dims=mmax)  
    hurst=nolds.hurst_rs(x)         
    if len(fnn[1][fnn[1]<=0.05])!=0: # 0.15 acum, 0.05 roll
        if 0 in fnn[1]:
            m=np.where(fnn[1]==0)[0][0]
        else:
            m=np.where(fnn[1]<=0.05)[0][0]
        lyapunov=nolds.lyap_e(x,emb_dim=2*(m)-1,matrix_dim=m,tau=lag)
        if sum(lyapunov)<0 and max(lyapunov)>0:
            caos='Deterministic Chaos'
        elif max(lyapunov)<0:
            caos='Stable' 
        else:
            caos='Random Motion' 
    else:
        caos='Not Found FNN'
        m=99
        lyapunov=np.array([99])
        fnn=np.array([99,99])
        hurst=99
    return lag,m,caos,lyapunov.sum(),lyapunov.max(),lyapunov,hurst,fnn[1].tolist()

def delay(x):
    # Returns the optimal Time-Delay from a time series
    # Use the autocorrelation function and the mutual information score
    da=0;dm=0
    n = len(x)
    y = x-x.mean()   
    r = np.correlate(y, y, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(y[:n-k]*y[-(n-k):]).sum() for k in range(n)]))
    auto = r/(x.var()*(np.arange(n, 0, -1)))
    while (auto[da]*auto[da+1])>0:
        da=da+1
    da=da+1    
    while sklearn.metrics.mutual_info_score(None,None,contingency=np.histogram2d(x, np.roll(x,dm),20)[0])>=sklearn.metrics.mutual_info_score(None,None, contingency=np.histogram2d(x, np.roll(x,dm+1),20)[0]):
        dm=dm+1
    lag=min(da,dm)+1
    return lag       

def global_false_nearest_neighbors(x, lag, min_dims=1, max_dims=15, **cutoffs):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    x = _vector(x)
    dimensions = np.arange(min_dims, max_dims + 1)
    false_neighbor_pcts = np.array([_gfnn(x, lag, n_dims, **cutoffs) for n_dims in dimensions])
    return dimensions, false_neighbor_pcts

def _gfnn(x, lag, n_dims, **cutoffs):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    # Global false nearest neighbors at a particular dimension.
    # Returns percent of all nearest neighbors that are still neighbors when the next dimension is unfolded.
    # Neighbors that can't be embedded due to lack of data are not counted in the denominator.
    offset = lag*n_dims
    is_true_neighbor = _is_true_neighbor(x, _radius(x), offset)
    return np.mean([
        not is_true_neighbor(indices, distance, **cutoffs)
        for indices, distance in _nearest_neighbors(reconstruct(x, lag, n_dims))
        if (indices + offset < x.size).all()
    ])

@jit
def yn_obj(yn_val,*args):
    # Objetive function for yn
    sta_sync1,ynr1,distanceA,indA,dist_gcm_gcm,s_max=args
    distanceB=np.sqrt((yn_val-sta_sync1)**2+ynr1)
    dist_sta_sta=min(distanceB)
    indB=np.where(distanceB==dist_sta_sta)
    if (distanceB[indA]*distanceA[indB])/(dist_sta_sta*dist_gcm_gcm)<=s_max:
        return distanceB[indA]+distanceA[indB]+dist_sta_sta
    else:
        return (distanceB[indA]+distanceA[indB]+dist_sta_sta)**2
        
@curry 
def _is_true_neighbor(x, attractor_radius, offset, indices, distance,relative_distance_cutoff=15, relative_radius_cutoff=2):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    distance_increase = np.abs(sub(*x[indices + offset])) 
    return (distance_increase / distance < relative_distance_cutoff and
            distance_increase / attractor_radius < relative_radius_cutoff)
 
@jit 
def _distance(y):
    distances, indices = sklearn.neighbors.NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(y).kneighbors(y)
    return distances, indices

def _nearest_neighbors(y):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    distances, indices = _distance(y)
    for distance, index in zip(distances, indices):
        yield index, distance[1]

@jit 
def _radius(x):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    return np.sqrt(((x - x.mean())**2).mean())

def reconstruct(x, lag, n_dims):
    # create the delayed vector from a time series
    x = _vector(x)
    lags = lag * np.arange(n_dims)
    return np.vstack(x[lag:lag - lags[-1] or None] for lag in lags).transpose()

def deconstruct(x, lag, n_dims):
    # create the time series from a delayed vector
    dec=np.empty(len(x)+lag*(n_dims-1))* np.nan
    dec[:len(x)]=x[:,0]
    dec[len(x):]=x[-lag*(n_dims-1):,-1]
    return dec

def _vector(x):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError('x(t) must be a 1-dimensional signal')
    return x   
      
def dateparse(x):
    # return pd.datetime.strptime(x,'%Y-%m-%d')
    return pd.datetime.strptime(x,'%d-%m-%Y')

def load_chaos(s1,caos_station,gcm_cell,cal_date,step):
    step_str_1=str(step)
    if s1[:3]!='GCM':  
        step=caos_station[2].loc[s1].dropna().dropna()
        # step_str_1=step[(step=='Deterministic Chaos').values].index[0]
        step_int_1=int(step_str_1)
        data_1=station[s1].loc[:cal_date]
        lag_1=caos_station[0][step_str_1].loc[s1]
        m_1=caos_station[1][step_str_1].loc[s1]
    else:
        start=dt.datetime(1989,1,1)
        step=caos_gcm[2].loc[int(s1[-1])].dropna()
        # step_str_1='5' # step[(step=='Deterministic Chaos').values].index[0] 
        step_int_1=int(step_str_1)
        data_1=model[int(s1[-1])]['GCM'].loc[start:cal_date] 
        lag_1=caos_gcm[0][step_str_1].loc[int(s1[-1])]
        m_1=caos_gcm[1][step_str_1].loc[int(s1[-1])]
    # ## Normalize Data
    # data_1=data_1/data_1.mean()   
    # Mean Rolling time series
    acum_1=data_1.rolling(window=step_int_1).mean().dropna()
    ## Normalize Data
    acum_1=acum_1/acum_1.mean()
    # Vector Reconstruction
    vec_1=reconstruct(acum_1, lag_1, m_1)
    return vec_1,m_1,lag_1

station_data='C:\\Users\\santi\\Google Drive\\Desktop\\Bogota_Sync\\Precipitacion.csv'
station_catalog='C:\\Users\\santi\\Google Drive\\Desktop\\Bogota_Sync\\Catalogo.csv'
model_historic='C:\\Users\\santi\\Google Drive\\Desktop\\Bogota_Sync\\HISTORIC\\'
output_folder='C:\\Users\\santi\\Google Drive\\Desktop\\Bogota_Sync\\Results\\'
int_acum=[3,5,7,15,30];
cal_date=dt.datetime(2019,12,31)

# station_data='C:\\Users\\santi\\Google Drive\\Desktop\\Test\\Precipitacion.csv'
# station_catalog='C:\\Users\\santi\\Google Drive\\Desktop\\Test\\Catalogo.csv'
# model_historic='C:\\Users\\santi\\Google Drive\\Desktop\\Test\\HISTORIC\\'
# model_rcp='C:\\Users\\santi\\Google Drive\\Desktop\\Test\\RCP45\\'
# output_folder='C:\\Users\\santi\\Google Drive\\Desktop\\Test\\Results\\'
# int_acum=[3,5,7,15,30];
# cal_date=dt.datetime(2005,12,31)
# val_date=dt.datetime(2010,12,31) 
# app_date=dt.datetime(2100,12,31)

# chaotic_statistcal_downscaling(station_data,station_catalog,model_historic,output_folder,int_acum,cal_date):
# station_data: csv file with the climatic data of the stations (fist column dates)
# station_catalog: csv file with latitude and longitude of the stations
# model_historic:  folder with the nc files with the historic experiment of the GCM
# output_folder: folder to save the results and plots
# int_acum: vector with the time steps to evaluate
# cal_date: last date of the calibration process
print("Synchronization Analysis")
print("Importing CSV Data")
station=pd.read_csv(station_data,index_col='Date',parse_dates=True) #, date_parser=dateparse)
catalogo=pd.read_csv(station_catalog,sep=',')
# catalogo=pd.read_csv(station_catalog,sep=';')
if any(catalogo['Longitud']<0):
    catalogo['Longitud']=catalogo['Longitud']+360
print("Loading NetCDF Data")
# Time model data
time_his=[];time_rcp=[];    
for filename in os.listdir(model_historic):
    his = nc.Dataset(model_historic+filename)
    if filename==os.listdir(model_historic)[0]:
        time_unit=his.variables['time'].units 
        time_cal=his.variables['time'].calendar
    time_valh=his.variables['time'][:] # Thi code works for daily values 
    time_his=np.concatenate((time_his,nc.num2date(time_valh,units=time_unit,calendar=time_cal)))    
time_his=[dt.datetime.strptime(k.strftime('%Y-%m-%d'),'%Y-%m-%d') for k in time_his]
# Latitude and Longitude Model 
gcm_lat=his.variables['lat'][:]
gcm_lon=his.variables['lon'][:]
gcm_lat=np.array([gcm_lat,]*1).flatten()
gcm_lon=np.array([gcm_lon,]*1).flatten() 
# Units Conversion
var_type=list(his.variables.keys())[-1]
if var_type=='pr':
    # Total Precipitation
    factor1=86400 # Conversion from kg/m^2/s to mm/day 
    factor2=0
    units='(mm)'
elif var_type=='tas' or var_type=='tas_min' or var_type=='tas_max':
    # Temperature
    factor1=1
    factor2=-273.15 # Conversion from °K to ° C
    units='(C)'     
else: 
    # Not Recognizable Units
    factor1=1
    factor2=0 
    units='NN'     
time_gcm=time_his
print("Get GCM cell and model information")
gcm_cell=pd.DataFrame(columns=['Lat','Lon','Model'],index=range(station.shape[1]))
model=[]
for z in catalogo.index:
    # Find positition of station in GCM cells   
    for cell_x in range(int(gcm_lon.shape[0])-1):
        if (gcm_lon[cell_x-1]+gcm_lon[cell_x])/2<=catalogo['Longitud'][z] and catalogo['Longitud'][z]<=[(gcm_lon[cell_x+1]+gcm_lon[cell_x])/2]:
            gcm_cell['Lon'][z]= cell_x   
            break                
    for cell_y in range(int(gcm_lat.shape[0])-1):  
        if (gcm_lat[cell_y-1]+gcm_lat[cell_y])/2<=catalogo['Latitud'][z] and catalogo['Latitud'][z]<=(gcm_lat[cell_y+1]+gcm_lat[cell_y])/2:                   
            gcm_cell['Lat'][z]= cell_y   
            break            
    # Check if GCM output already read in model
    if ((gcm_cell[['Lon','Lat']].loc[z] == gcm_cell[['Lon','Lat']][:z]).any()).all():   
        unique=gcm_cell[gcm_cell[['Lon','Lat']].loc[z] == gcm_cell[['Lon','Lat']][:z]].dropna(subset=['Lon','Lat'])
        gcm_cell['Model'][z]=gcm_cell['Model'].loc[unique.index[0]]
    else:         
        data_his=[];data_rcp=[];
        for filename in os.listdir(model_historic):
            his = nc.Dataset(model_historic+'\\'+filename)    
            data_his=np.concatenate((data_his,his.variables[var_type][:,int(gcm_cell['Lat'][z]),int(gcm_cell['Lon'][z])]*factor1+factor2))    
            model_df=pd.DataFrame(data_his,index=time_gcm,columns=['GCM'])
        model.append(model_df)
        gcm_cell['Model'][z]=len(model) 
his.close
gcm_cell['Lat']=[gcm_lat[gcm_cell['Lat'][i]] for i in range(gcm_cell['Lat'].shape[0])]
gcm_cell['Lon']=[gcm_lon[gcm_cell['Lon'][j]] for j in range(gcm_cell['Lon'].shape[0])]
gcm_cell.to_csv(output_folder+'GCM_cells.csv')
################################ DYNAMICAL ANALYSIS  ######################################
## MODES:
## Mode 1 compute only for the step needed for the downscaling
## Mode 2 computes all dynamical system properties for all the steps
## PROPERTIES:
## 0: Time Delay
## 1: Embedding Dimension
## 2: ¿Deterministic Chaos?
## 3: Lyapunov Sum
## 4: Lyapunov Max
## 5: Lyapunov Spectrum
## 6: Hurst Exponent
## 7: False Nearest Neighbors      
##################### Create a file for each dynamics property ##############################
properties=['Time_delay','Dimension','Chaos','Lyapunov_sum','Lyapunov_max','Lyapunov','Hurst','FNN']
prop=len(properties)
caos_gcm=[]
caos_station=[]
print("Dynamics Analysis:")
### Check if dynamics analysis already exist:
if os.path.exists(output_folder+'GCM_'+properties[0]+'.csv'):
    for i in range(prop):                   
        caos_gcm.append(pd.read_csv(output_folder+'GCM_'+properties[i]+'.csv',index_col='ID'))
        caos_station.append(pd.read_csv(output_folder+'local_'+properties[i]+'.csv',index_col='ID'))
else:    
    mode=2
    for i in range(prop): # Number of Dynamics Properties to analyze
        caos_gcm.append(pd.DataFrame(columns=int_acum,index=range(len(model))))
        caos_station.append(pd.DataFrame(columns=int_acum,index=catalogo['ID'])) 
    for z in catalogo.index: 
        s=str(catalogo['ID'][z])
        data_sta=station[s].loc[:cal_date]
        data_sta=data_sta.dropna()
        print("  Station "+str(z+1) +": " +s)
        m=gcm_cell['Model'][z]-1
        data_gcm=model[m].loc[:cal_date]   
        print('    GCM: '+str(m+1)+' - Lat: '+'{:.4f}'.format(gcm_cell['Lat'][z])+'- Lon: '+'{:.4f}'.format(gcm_cell['Lon'][z]))
        for step in int_acum: #[1:2]
            print("        step: " + str(step))      
            if not isinstance(caos_gcm[2][step].loc[m], str):
                print("          GCM doesn't exist ... ")
                if not isinstance(caos_station[2][step][s], str):
                    print("          Station ... analyzing ...")
                    pr_sta=data_sta.rolling(window=step).sum().dropna() # data_sta.resample(str(step)+'D').mean()
                    dynamics=dynamics_analysis(pr_sta.values)           
                    for i in range(prop):
                        caos_station[i][step].loc[s]=dynamics[i]
                if caos_station[2][step].loc[s]=='Deterministic Chaos':
                    print("          Station Deterministic Chaos ...")
                    print("          GCM ... analyzing ...")
                    pr_gcm=data_gcm.rolling(window=step).sum().dropna() # data_gcm.resample(str(step)+'D').mean()
                    dynamics=dynamics_analysis(pr_gcm.values)
                    for i in range(prop):
                        caos_gcm[i][step].loc[m]=dynamics[i]
                    if caos_gcm[2][step].loc[m]=='Deterministic Chaos':
                        print("          GCM ... Deterministic Chaos ...")
                        if mode==1:
                            break
                    else:
                        print("          GCM not Deterministic Chaos")     
                else:
                    print("          Station not Deterministic Chaos") 
                    print("          GCM ... analyzing ...")
                    if mode==2:
                        pr_gcm=data_gcm.rolling(window=step).sum().dropna() # data_gcm.resample(str(step)+'D').mean()
                        dynamics=dynamics_analysis(pr_gcm.values)
                        for i in range(prop):
                            caos_gcm[i][step].loc[m]=dynamics[i]
                        if caos_gcm[2][step].loc[m]=='Deterministic Chaos':
                            print("          GCM ... Deterministic Chaos ...") 
            elif caos_gcm[2][step].loc[m]== 'Deterministic Chaos':
                print("          GCM exist ... analyzing station...")
                if not isinstance(caos_station[2][step][s], str):
                    pr_sta=data_sta.rolling(window=step).sum().dropna() # data_sta.resample(str(step)+'D').mean() #
                    dynamics=dynamics_analysis(pr_sta.values)           
                    for i in range(prop):
                        caos_station[i][step].loc[s]=dynamics[i]    
                if caos_station[2][step].loc[s]=='Deterministic Chaos':
                    print("          Station Deterministic Chaos") 
                    if mode==1:
                        break
                else:
                    print("          Station not Deterministic Chaos") 
            else:
                print("          GCM not Deterministic Chaos") 
                if mode==2:
                    print("          Station ... analyzing ...")
                    pr_sta=data_sta.rolling(window=step).sum().dropna() # data_sta.resample(str(step)+'D').mean()
                    dynamics=dynamics_analysis(pr_sta.values)           
                    for i in range(prop):
                        caos_station[i][step].loc[s]=dynamics[i]  
                    if caos_station[2][step].loc[s]=='Deterministic Chaos':
                        print("          Station Deterministic Chaos ...")
    # Write Dynamical System Properties to CSV files
    for i in range(prop):                   
        caos_gcm[i].to_csv(output_folder+'GCM_'+properties[i]+'.csv',index_label='ID')
        caos_station[i].to_csv(output_folder+'local_'+properties[i]+'.csv')
################################ SYNCHRONIZATION ANALYSIS  ######################################
a=time.time()      
step=7
# Select elements to synchronize
station_selected=caos_station[0][caos_station[2][str(step)]!='Random Motion'].index.tolist()
gcm_selected=caos_gcm[0][caos_gcm[2][str(step)]!='Random Motion'].index.tolist()
# Sync Results Variable
names=station_selected+['GCM'+str(gcm_selected[i]) for i in range(len(gcm_selected))]
sync=pd.DataFrame(columns=names,index=names)
for i in names:
    sync[i][i]=1
# Pre-allocate results
len_cal=station[:cal_date].shape[0]
len_names=len(names)
mfnnYYY=np.empty((len_cal,2,len_names))
mfnnYYY[:]=np.nan
## Save ynn distances and ynnr position
print("Loading Distances ...")
list_distance=[]
for s1 in names: 
    vec_1,m_1,lag_1=load_chaos(s1,caos_station,gcm_cell,cal_date,step)     
    distYYY=sklearn.metrics.pairwise.euclidean_distances(vec_1)
    list_distance.append(distYYY)
    for t in range(1,vec_1.shape[0]):
        # Distance in Driver
        distanceB1=distYYY[:t,t]
        distanceB1=np.where(distanceB1==0, distanceB1.max(), distanceB1) # replace for 9999 or  distanceB1.max()
        mfnnYYY[t,0,names.index(s1)]= min(distanceB1)
        indB=np.argmin(distanceB1)
        mfnnYYY[t,1,names.index(s1)]=indB
### Calculate all distances
print("Calculate Sync MFNN ...")
for s1 in tqdm(names):  
    print("Element: "+s1)
    ## First Time Series   
    x=mfnnYYY[:len_cal,1,names.index(s1)]
    x = x[~np.isnan(x)]  
    s2_list=sync[s1].drop(s1)
    ##########################
    # vec_1,m_1,lag_1=load_chaos(s1,caos_station,gcm_cell,cal_date,step)     
    # distZZZ=sklearn.metrics.pairwise.euclidean_distances(vec_1)
    pos_1=names.index(s1)
    distZZZ=list_distance[pos_1]
    ##########################   
    for s2 in s2_list.index:
        ### Second Time Series
        if ~np.isnan(sync[s1][s2]):
            continue
        y=mfnnYYY[:len_cal,1,names.index(s2)]
        y = y[~np.isnan(y)]  
        lenxy=min(len(x),len(y))
        mfnn=pd.DataFrame(index=range(lenxy),columns=['Yn-YnnR','Xn-XnnD','Yn-YnnD','Xn-XnnR','nnD','nnR','S','U']) 
        mfnn['Yn-YnnR']=mfnnYYY[:lenxy,0,names.index(s1)]
        mfnn['nnR']=mfnnYYY[:lenxy,1,names.index(s1)]
        mfnn['Xn-XnnD']=mfnnYYY[:lenxy,0,names.index(s2)]
        mfnn['nnD']=mfnnYYY[:lenxy,1,names.index(s2)]
        ##########################
        # vec_2,m_2,lag_2=load_chaos(s2,caos_station,gcm_cell,cal_date,step)     
        # distXXX=sklearn.metrics.pairwise.euclidean_distances(vec_2)
        pos_2=names.index(s2)
        distXXX=list_distance[pos_2]
        ########################## 
        ## Parameters u and ss: Mutual False Nearest Neighbor" 
        mfnn['Yn-YnnD'].loc[1:]=distZZZ[range(1,lenxy),mfnn['nnD'].values[1:].astype(int)]
        mfnn['Xn-XnnR'].loc[1:]=distXXX[range(1,lenxy),mfnn['nnR'].values[1:].astype(int)]      
        ## Compute ss and u
        mfnn['S']=(mfnn['Xn-XnnR']*mfnn['Yn-YnnD'])/(mfnn['Xn-XnnD']*mfnn['Yn-YnnR'])
        mfnn['U'].loc[1:]=mfnn['S'].loc[1:].cumsum()/mfnn.index[1:]
        ## Write Results
        len_u=int(0.8*mfnn['U'].shape[0]) 
        sync[s1][s2]=mfnn['U'][len_u:].mean() 
        sync[s2][s1]=sync[s1][s2]
sync.to_csv(output_folder+'Sync_'+str(step)+'_steps.csv')
