#%%
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
from inlog import Input
#%%
outdir=Path("/project/meteo/work/Paul.Ockenfuss/PhD/Experiments/RadarAdvectionSimulator")
regular_file=Path("/scratch/p/Paul.Ockenfuss/PhD/Experiments/KneifelRetrieval/Cloudnet/2022/202203_joyrad35_regular.nc")
regular=xr.open_dataset(regular_file)
#%%

regular=regular.sel(time=slice("2022-03-30 21:00", "2022-03-30 23:00"))
v=regular.vwind.mean('time') #Vectorial mean by averaging components
u=regular.uwind.mean('time')
vv=np.sqrt(v**2+u**2)
w=regular.v.mean('time') #doppler velocity as vertical velocity
w=w.interpolate_na('height')
ds=regular.Z.rename("radar").astype(float).to_dataset()
ds["v"]=vv
ds["w"]=w
ds=ds.sel(height=slice(1500, 4000))


# %%
outfile=outdir/f"20220330_rain.nc"
ds.to_netcdf(outfile)
inp=Input.Input(None, "1.0")
inp.add_outfile(outfile)
inp.write_log(outfile, [regular_file], '.log')

