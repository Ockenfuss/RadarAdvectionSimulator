#%%
import numpy as np
import xarray as xr
from luti.xarray import invert_data_array
from luti import LinearInterpolator, BoxChecker
import matplotlib.pyplot as plt
#%%
#testing
# radar=xr.DataArray(np.ones((30,30)),coords=[('height', np.arange(30)), ('time', np.arange(30))] )
# v=xr.DataArray(np.linspace(0.5,3,len(radar.height)), coords=[radar.height])
# w=0.0*v-0.5
# min_dist=10
# nt=1000
# sn=100
# height=v.height
# vmax=float(abs(v).max())
# vmin=float(abs(v).min())
# deltat=radar.time.max()-radar.time.min()
# pos=xr.DataArray(np.linspace(-vmax*deltat, vmax*deltat, 200), dims=["pos"])
# pos.coords["pos"]=("pos", pos.values)
#%%

#%%

def drop_nondimensional(da):
    da=da.drop([d for d in list(da.coords.keys()) if d not in da.dims])#Drop all non-dimensional coordinates
    return da

# height: define points where we want the trajectory
#v[height], w[height]: velocity field
#Since this is a recursive problem, we need an explicit loop, there is no way to do this purely in xarray/numpy
#min_dist: minimum horizontal distance until which the particles are traced
#nt: Number of integration steps. The deltat is chosen such that the slowest parcel reaches the radar after nt steps.
def get_trajectory(height, v, w, tmin, tmax, xstart,tstart=0.0, nt=1000):
    assert(np.all(abs(v)>1e-5)) #no zero speed
    assert(np.all(v/v[0]>0)) #no sign change
    assert(tmin<tstart)
    assert(tmax>tstart)
    vmin=abs(v).min().values
    dt=(tmax-tmin)/nt
    time=xr.DataArray(np.linspace(tmin, tmax, nt), dims=['time'])
    time.coords['time']=('time', time.values)
    Y=height+0.0*time #just to get a new, broadcasted array
    X=0*Y+xstart
    i_start=int((tstart-tmin)/(tmax-tmin)*nt)
    for i in range(i_start+1,nt):
        X[:,i]=X[:,i-1]+v.interp(height=Y[:,i-1], kwargs={"fill_value":(v[0], v[-1])})*dt
        Y[:,i]=Y[:,i-1]+w.interp(height=Y[:,i-1], kwargs={"fill_value":(w[0], w[-1])})*dt
    for i in range(i_start-1,-1,-1):
        X[:,i]=X[:,i+1]-v.interp(height=Y[:,i+1], kwargs={"fill_value":(v[0], v[-1])})*dt
        Y[:,i]=Y[:,i+1]-w.interp(height=Y[:,i+1], kwargs={"fill_value":(w[0], w[-1])})*dt
    return X,Y

#%%
def get_time_from_radar(height, pos, v,w, radar_pos=0.0):
    vmin=abs(v).min().values
    tmin=((pos.min()-radar_pos)/vmin).values #time the slowest parcel would need to reach the radar (forward)
    tmax=((pos.max()-radar_pos)/vmin).values #time the slowest parcel would neet to reach the radar (backward)
    X,Y=get_trajectory(height, v,w,tmin, tmax, radar_pos) #X[y,t], Y[y,t]
    X=X.expand_dims({"out":["pos"]})
    T=invert_data_array(X, input_params=["time"], output_dim='out', output_grid={"pos":pos.values}).squeeze()
    # T=T.rename({"height":"T_height", "pos":"T_pos"})
    x_pos=pos+0.0*height
    y_pos=0.0*pos+height
    time_from_radar=T.interp(pos=x_pos, height=y_pos)
    time_from_radar=drop_nondimensional(time_from_radar)
    height_at_radar=Y.interp(time=-time_from_radar, height=y_pos)
    height_at_radar=drop_nondimensional(height_at_radar)
    return time_from_radar, height_at_radar
#%%

def get_cloudfield(radar, v,w,sn=100):
    v_radar=v.interp(height=radar.height)
    w_radar=w.interp(height=radar.height)
    vmax=float(abs(v_radar).max())
    wmax=float(abs(w_radar).max())
    deltat=radar.time.max()-radar.time.min()
    pos=xr.DataArray(np.linspace(-vmax*deltat, vmax*deltat, sn), dims=["pos"])
    pos.coords["pos"]=pos
    # hmax=-abs(w).max()*pos.min()/abs(v).min()+radar.height.max() #How far would the horizontally slowest particle fall until it reaches the radar?
    # hmin=radar.height.min()-abs(w).max()*pos.max()/abs(v).min() #How far will the horizontally slowest particle fall 
    hmax=abs(w).max()*deltat+radar.height.max()
    hmin=max(radar.height.min()-abs(w).max()*deltat, 0.0) #Do not track below ground in general
    height=xr.DataArray(np.linspace(hmin, hmax, sn), dims=["height"])
    height.coords["height"]=("height", height.values)
    time_at_radar, height_at_radar=get_time_from_radar(height, pos, v, w)
    time_at_radar=-time_at_radar+radar.time
    height_at_radar=height_at_radar.expand_dims({'time':radar.time})
    # return time_at_radar, height_at_radar
    cloudfield=radar.interp(time=time_at_radar, height=height_at_radar)
    return cloudfield











# %%
