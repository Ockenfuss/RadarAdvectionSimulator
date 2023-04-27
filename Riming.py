#%%
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
from inlog import Input
inp=Input.Input(None, "1.0")
outdir=Path("/project/meteo/work/Paul.Ockenfuss/PhD/Experiments/RadarAdvectionSimulator/Data")
#%%
regular_file=Path("/scratch/p/Paul.Ockenfuss/PhD/Experiments/KneifelRetrieval/Cloudnet/2018/201803_joyrad35_regular.nc")
retrieval_file=Path("/home/p/Paul.Ockenfuss/Work/PhD/Experiments/KneifelRetrieval/Output/Joyrad35/Rolling/2018/201803_joyrad35_retrieval_rolling.nc")
fr=xr.open_dataset(retrieval_file).fr.drop('band')
fr=fr.where(fr>0.6)
regular=xr.open_dataset(regular_file)
tablefile=Path("/project/meteo/work/Paul.Ockenfuss/PhD/Experiments/KneifelRetrieval/Output/Joyrad35/Rolling/EventTable.hdf")
df=pd.read_hdf(tablefile, "df")
#%%

day="2018-03-12"
df=df.loc[df.starttime.dt.date==datetime.strptime(day, "%Y-%m-%d").date()]
df=df.iloc[0]
regular=regular.sel(time=slice(df.starttime, df.stoptime))
fr=fr.sel(time=slice(df.starttime, df.stoptime))
v=regular.vwind.mean('time') #Vectorial mean by averaging components
u=regular.uwind.mean('time')
vv=np.sqrt(v**2+u**2)
ds=fr.rename("radar").to_dataset()
ds["v"]=vv
ds=ds.sel(height=slice(0, df.ev_height+df.ev_depth))


# %%
outfile=outdir/f"20180312_event.nc"
ds.to_netcdf(outfile)
inp.set_outfile(outfile)
inp.write_log(outfile, [retrieval_file], '.log')

