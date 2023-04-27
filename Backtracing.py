#%%
# Full tracing of praticles in a 3D field
import xarray as xr
import numpy as np
from pathlib import Path
import advesi as adv
import datetime as dt
import pandas as pd
import phdpy.spectra as psp
#%% Test with real data
workdir=Path("/project/meteo/work/Paul.Ockenfuss/PhD/Experiments/RadarAdvectionSimulator/Data")
anidir=Path("/project/meteo/work/Paul.Ockenfuss/PhD/Experiments/RadarAdvectionSimulator/Movies")
data_full=xr.open_dataset(workdir/"20211210_mahnwindfield.nc")
data_full=data_full.sel(swipe_time="2021-12-10 10:15:00", method='nearest')
swipe_time=data_full.swipe_time.values
data_full=data_full.drop('swipe_time')
data_full=data_full.rename(height='z')
data=data_full.sel(z=slice(1000,2000))
w_doppler=data.radar
w_doppler=w_doppler.where(w_doppler!=w_doppler.not_detectable)
w_doppler=w_doppler.rename(time='t')
u=data.u
v=data.v
#%% Alternative: Use raw xband data
data_raw=xr.open_dataset("/archive/meteo/external-obs/juelich/joyrad10/2021/12/10/20211210_100006.znc", decode_cf=False)[["VELg", "npw1", "SPCco", "HSDco", "SNRCorFaCo", "RadarConst"]].load()
data_raw.coords['doppler']=('doppler', -1*data_raw.doppler.values)
data_raw=data_raw.sortby('doppler').sel(doppler=slice(-5,3))
data_raw.time.attrs['units']=data_raw.time.long_name
data_raw=xr.decode_cf(data_raw)
data_raw=data_raw.sel(time=slice(swipe_time-np.timedelta64(15, 'm'), swipe_time+np.timedelta64(15, 'm')))
data_raw=data_raw.isel(time=slice(0,300))
data_raw=data_raw.sel(range=slice(1000,2000))
data_raw.coords["time"]=('time', ((data_raw.time-data_raw.time[0])/1e9).astype(float).values)
data_raw=xr.merge([psp.process_spectra(data_raw, 'o'), data_raw])
data_raw=data_raw.rename(range='z', time='t')

radarfield=data_raw.VELg
# field=w_doppler
plotargs={'vmin':-3, 'vmax':0, 'cmap':'jet'}
# field=np.log(data_raw.Zg)
# plotargs={'vmin':-6, 'vmax':6, 'cmap':'coolwarm'}
#%% Split spectra
def split_spectra(v, w):
    w_tot=w.sum(dim='doppler')
    cumfun=w.cumsum('doppler')
    v1=v.weighted(w.where(cumfun<0.333*w_tot, other=0.0)).mean('doppler')
    v2=v.weighted(w.where(np.logical_and(cumfun>=0.333*w_tot, cumfun<0.666*w_tot), other=0.0)).mean('doppler')
    v3=v.weighted(w.where(cumfun>=0.666*w_tot, other=0.0)).mean('doppler')
    w1=w.where(cumfun<0.333*w_tot, other=0.0).sum('doppler')
    w2=w.where(np.logical_and(cumfun>=0.333*w_tot, cumfun<0.666*w_tot), other=0.0).sum('doppler')
    w3=w.where(cumfun>=0.666*w_tot, other=0.0).sum('doppler')
    return xr.concat([v1, v2, v3], dim='doppler'), xr.concat([w1, w2, w3], dim='doppler')
weights=data_raw.SPCco_cal.fillna(0.0)
w_doppler=data_raw.doppler.broadcast_like(data_raw.SPCco_cal)
w_doppler,weights=split_spectra(w_doppler, weights)
w_doppler=w_doppler.where(weights.notnull()) #just be sure that if there are nans in the velocity, the weights are nan as well
weights=weights.where(w_doppler.notnull())
weighted_field=w_doppler*weights
#%% Test: Plot spectra
# fig, ax=plt.subplots()
# it=30
# iz=10
# data_raw.SPCco_cal.isel(t=it, z=iz).plot(ax=ax)
# # ax.axvline(data_raw.VELg.isel(t=it, z=iz).item())
# # ax.axvline(psp.spectrum_mean_velocity(data_raw.SPCco_cal.isel(t=it, z=iz)).item())
# ax.axvline(v.isel(doppler=0).isel(t=it, z=iz).item())
# ax.axvline(v.isel(doppler=1).isel(t=it, z=iz).item())
# ax.axvline(v.isel(doppler=2).isel(t=it, z=iz).item())

# %%
w=np.linspace(w_doppler.min(), w_doppler.max(), 100)
w=xr.DataArray(w, coords=[('w', w)])
ff=adv.FlowField_Collection(u,v,w)
#%%Calculate trajectories
traj_coll=adv.Trajectory_Collection.from_flowfield(ff, x0=0.0, y0=0.0, z0=w_doppler.z.values, dt=1.0, steps=1000, steps_backward=1000, savesteps=10, interp=False)

#%%
#Create particles
weighted_field=weighted_field.expand_dims(x=[0.0],y=[0.0])
w_doppler=w_doppler.broadcast_like(weighted_field)
weights=weights.broadcast_like(weighted_field)
field_coll=adv.Field_Collection(weighted_field)
weights_coll=adv.Field_Collection(weights)
#Create particle collection from field
part_coll_field=adv.Particle_Collection.from_field_collection(field_coll, field_selectors={'w':w_doppler})
part_coll_weight=adv.Particle_Collection.from_field_collection(weights_coll, field_selectors={'w':w_doppler})
#%%
#Create paths using trajectories
# output_times=np.linspace(field_coll.f.t.min().item(), field_coll.f.t.max().item(), 200)
output_times=field_coll.f.t.values
path_coll=adv.Path_Collection.from_particle_collection(part_coll_field, traj_coll,output_times)
#%%
#Create desired output field
size_x=5000
size_y=10000
aggregation_method='sum'
output_grid_field=adv.Field_Collection.create_regular(output_times, [-size_x, size_x], [-size_y,size_y], [1000,2000], 200, 200, 50)
output_grid_weight=adv.Field_Collection.create_regular(output_times, [-size_x, size_x], [-size_y,size_y], [1000,2000], 200, 200, 50)
output_grid_field.fill_with(part_coll_field, path_coll, aggregation=aggregation_method)
output_grid_weight.fill_with(part_coll_weight, path_coll, aggregation=aggregation_method)
f=output_grid_field.f#/
f=output_grid_weight.f
#%% Crop field
nonempty_x=f.notnull().sum(['y', 'z', 't'])>0
nonempty_y=f.notnull().sum(['x', 'z', 't'])>0
f=f.isel(x=slice(nonempty_x.argmax().item(), nonempty_x.cumsum().argmax().item()))
f=f.isel(y=slice(nonempty_y.argmax().item(), nonempty_y.cumsum().argmax().item()))
#%% Save the field
# outfile=workdir/f"{pd.to_datetime(swipe_time).strftime('%Y%m%d_%H%M%S')}_advesi_{radarfield.name}_{aggregation_method}.nc"
# f.to_netcdf(outfile)
#%% Calculate wind direction
from metpy.calc import wind_direction
from metpy.units import units
theta=wind_direction(u*units('m/s'),v*units('m/s'))
theta=np.deg2rad(theta.mean())
windspeed_proj=u*np.sin(theta)+v*np.cos(theta)
#%% Project to tilted plane
rad=f.x*np.sin(theta)+f.y*np.cos(theta)
f.coords["rad"]=(['x', 'y'], rad.values)
bins=np.linspace(-10000.0,10000.0,400)
frad=f.groupby_bins('rad', bins=bins, labels=bins[:-1]).mean().rename(rad_bins='rad')
#crop
nonempty=frad.notnull().sum(['z', 't'])>100
frad=frad.isel(rad=slice(nonempty.argmax().item(), nonempty.cumsum().argmax().item()))
#interpolate
# frad=frad.interpolate_na(dim='rad')
#%% Plot plane view
aspect=(frad.rad.max()-frad.rad.min())/(frad.z.max()-frad.z.min())
frad.isel(t=100).plot(x='rad', size=3, aspect=10)#, **plotargs)
#%% topview
topview=f.mean('z')
topview.isel(t=100).plot(x='x', **plotargs)
#%% Radar view
radarfield.squeeze().plot(x='t', size=3, aspect=6, **plotargs)
#%% Animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def make_animation(radar, cloudfield, topview, cloudaspect=1.0, imargs={}):
    deltat=radar.time.max()-radar.time.min()
    radar_plotfield=xr.DataArray(np.zeros((len(radar.z), len(radar.time)*2)), coords=[("z", radar.z.values), ("reltime", np.linspace(deltat, -deltat, 2*len(radar.time)))])
    radar_plotfield[:,len(radar.time):]=radar.transpose("z", "time").values
    fig = plt.figure(figsize=(14,8))
    grid = plt.GridSpec(2, 2, width_ratios=[4,1])#Define grid and specify over how many cells axes spread
    ax1 = fig.add_subplot(grid[0,0])
    ax2 = fig.add_subplot(grid[1,0])
    ax3 = fig.add_subplot(grid[0,1])
    im1=ax1.imshow(cloudfield.isel(time=0).transpose("z", "pos").values, origin='lower', extent=(cloudfield.pos.min(), cloudfield.pos.max(), cloudfield.z.min(), cloudfield.z.max()), aspect=cloudaspect, **imargs)
    ax1.axvline(0, color='red')
    ax1.set_xlabel("Horizontal Distance to Radar")
    ax1.set_ylabel("Height")
    ax1.set_title('View from the side')
    im2=ax2.imshow(radar_plotfield.transpose("z", "reltime").values, origin='lower', extent=(radar_plotfield.reltime[0], radar_plotfield.reltime[-1], radar_plotfield.z.min(), radar_plotfield.z.max()), **imargs)
    ax2.axvline(0, color='red')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Height")
    ax2.set_title('Radar measurement')
    im3=ax3.imshow(topview.isel(time=0).transpose("y", "x").values, origin='lower', extent=(topview.x.min(), topview.x.max(), topview.y.min(), topview.y.max()), **imargs)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_title('View from top')

    def animate(i):
        im1.set_array(cloudfield.isel(time=i).transpose("z", "pos").values)
        im2.set_array(radar_plotfield.roll(reltime=-i).transpose("z", "reltime").values)
        im3.set_array(topview.isel(time=i).transpose("y", "x").values)
        return [im1, im2, im3]

    anitimes=range(len(cloudfield.time))
    if len(anitimes)>400:
        anitimes=np.linspace(0,len(anitimes)-1, 400).astype(int)
    ani=FuncAnimation(fig, animate, anitimes, blit=True, interval=100)
    return ani
ani=make_animation(radarfield.rename(t='time'), frad.rename(t='time', rad='pos'),topview.rename(t='time'),cloudaspect=2.0, imargs=plotargs)

# %%
anifile=anidir/f'{pd.to_datetime(swipe_time).strftime("%Y%m%d_%H%M%S")}_advesi_{radarfield.name}_{aggregation_method}.mp4'
ani.save(anifile, progress_callback =lambda i, n: print(f'Saving frame {i} of {n}'))

# %%
