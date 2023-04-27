#%%
# Create a dataset with xband birdbath data and the wind vectors from Jan Schween PPI fits
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from inlog import Input
inp=Input.Input(None, "1.0")
import datatree as xt
outdir=Path("/project/meteo/work/Paul.Ockenfuss/PhD/Experiments/RadarAdvectionSimulator/Data")
regfile="/scratch/p/Paul.Ockenfuss/PhD/Experiments/KneifelRetrieval/Joyrad10/2021/202112_joyrad10_regular.nc"
windppi="/archive/meteo/external-obs/juelich/joyrad35/wind_ppi/data/2021/12/20211210_joyrad35_wind_profile.nc"
#%% Find the ideal times, where the radar was pointing up during a scan
correctdir=Path("/home/p/Paul.Ockenfuss/Scratch/PhD/Experiments/RHIWindCorrection/Fit_Correction")
rhifiles=list(correctdir.glob("20211210*rhi_fitcorrected.nc"))
swipe_times=[]
for rhifile in rhifiles:
    ds=xr.open_dataset(rhifile)
    elev_first=ds.elev.isel(time=slice(None, ds.elev.argmax().values))
    elev_second=ds.elev.isel(time=slice(ds.elev.argmax().values, None))
    swipe_times.append(elev_first.time.isel(time=abs(elev_first-np.deg2rad(90.0)).argmin()).values)
    swipe_times.append(elev_second.time.isel(time=abs(elev_second-np.deg2rad(90.0)).argmin()).values)
swipe_times=sorted(swipe_times)
#%% open x band data and cut out sections for the scans from the Ka band radar
regds=xr.open_dataset(regfile)
regds=regds.sel(height=slice(0,5000))
datasets=[]
for s in swipe_times:
    ds=regds.sel(time=slice(s-np.timedelta64(15, 'm'), s+np.timedelta64(15, 'm')))
    assert(len(ds.time)==60)
    ds=ds.v.rename("radar").astype(float).to_dataset()
    ds["time_abs"]=ds.time
    ds["time"]=('time', ((ds.time-ds.time[0])/1e9).astype(float).values)
    datasets.append(ds)
ds=xr.concat(datasets, "swipe_time")
ds["swipe_time"]=('swipe_time', swipe_times)
ds.swipe_time.attrs["description"]='The time where the radar was pointing vertically while doing a RHI scan swipe.'
#%% Select corresponding wind vertical profiles for each swipe
mahnppidir=Path("/archive/meteo/external-obs/juelich/joyrad35/wind_ppi/data/2021/12/20211210_joyrad35_wind_profile.nc")
mahnppi=xr.open_dataset(mahnppidir)[["wind_vec"]]
mahnppi.coords['time']=('time', (mahnppi.time.values-2440587.5)*86400)
mahnppi.time.attrs['units']='seconds since 1970-01-01'
mahnppi=xr.decode_cf(mahnppi)
mahnppi=mahnppi.astype(float)

mahnppi=mahnppi.sel(time=ds.swipe_time, method='nearest').drop('time')
u=mahnppi.wind_vec.isel(N_vec=0)
u=u.interp(height=ds.height)
u=u.interpolate_na('height', fill_value='extrapolate')
v=mahnppi.wind_vec.isel(N_vec=1)
v=v.interp(height=ds.height)
v=v.interpolate_na('height', fill_value='extrapolate')
vh=np.sqrt(v**2+u**2)
ds["vh"]=vh
ds["u"]=u
ds["v"]=v
#%% Save
outfile=outdir/f"20211210_mahnwindfield.nc"
ds.to_netcdf(outfile)
inp.set_outfile(outfile)
inp.write_log(outfile, [regfile], '.log')

# %%