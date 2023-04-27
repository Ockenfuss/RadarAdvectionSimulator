#%%
import matplotlib as mpl
# mpl.use("Agg")#specify backend, before importing pyplot! Important if Display variable is not set. Alternative: TkAgg
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import sys
from scipy.signal import gaussian
from Field import get_cloudfield
workdir=Path("/project/meteo/work/Paul.Ockenfuss/PhD/Experiments/RadarAdvectionSimulator/Data")
outdir=Path("/project/meteo/work/Paul.Ockenfuss/PhD/Experiments/RadarAdvectionSimulator/Output")
anidir=Path("/project/meteo/work/Paul.Ockenfuss/PhD/Experiments/RadarAdvectionSimulator/Movies")


#%%
def retrieve_cloudfield(radar,v, sn=100, plimit=None):
    v_radar=v.interp(height=radar.height)
    vmax=float(abs(v_radar).max())
    deltat=radar.time.max()-radar.time.min()
    if plimit is None:
        plimit=(-vmax*deltat, vmax*deltat)
    pos=xr.DataArray(np.linspace(plimit[0], plimit[1], sn), dims=["pos"])
    pos.coords["pos"]=pos
    time_to_radar=-pos/v_radar+radar.time #positive if before radar, negative if after radar
    height=radar.height.expand_dims({'time':radar.time, 'pos':pos})
    cloudfield=radar.interp(time=time_to_radar, height=height)
    return cloudfield

def get_extent(radar_mask, v):
    Tr=radar_mask.time[radar_mask.argmax('time')].drop('time')
    Tr=Tr.where(radar_mask.sum('time')>0)
    Tl=radar_mask.time[radar_mask.cumsum('time').argmax('time')].drop('time')
    Tl=Tl.where(radar_mask.sum('time')>0)
    Pl=v*(radar_mask.time-Tl)#positive positions if point is past radar, i.e. time>Tl
    Pr=v*(radar_mask.time-Tr)
    return Pl, Pr


def radar_from_image(filepath):
    image=imread(filepath)
    # image=image[::-1,:,3]
    image=image[::-1,:].mean(axis=2)
    radar=xr.DataArray(image, coords=[('height', np.arange(image.shape[0])), ('time', np.arange(image.shape[1]))])
    # radar=radar.where(radar>0.0)
    return radar

def plot_vertical_line(x,y,ax4):
    ax4.plot([x,x], [y,y], [0,250], color='red')

def plot_xz_plane(y, ax4):
    xx=np.array([[0,100], [0,100]])
    yy=np.array([[y,y], [y,y]])
    z=np.array([[200,200], [0,0]])
    ax4.plot_surface(xx,yy,z, color='red', alpha=0.5)

def plot_3d_field(cloudfield, ax):
    verts, faces, normals, values = measure.marching_cubes(cloudfield.fillna(0.0).transpose("time", "pos", "height").values, 0, step_size=2)
    #3D Plot
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('lightgrey')
    # mesh.set_facecolor('blue')
    # mesh.set_alpha(0.3)
    ax.add_collection3d(mesh)

    # plot_vertical_line(100,50,ax)
    # plot_xz_plane(20, ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_xticklabels([])

    ax.set_xlim(verts[:,0].min(), verts[:,0].max())  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(verts[:,1].min(), verts[:,1].max())  # a = 6 (times two for 2nd ellipsoid)
    ax.set_zlim(verts[:,2].min(), verts[:,2].max())  # a = 6 (times two for 2nd ellipsoid)

#%%
def make_animation(radar, cloudfield, v,w=None, cloudaspect=1.0, imargs={}):
    deltat=radar.time.max()-radar.time.min()
    radar_plotfield=xr.DataArray(np.zeros((len(radar.height), len(radar.time)*2)), coords=[("height", radar.height.values), ("reltime", np.linspace(deltat, -deltat, 2*len(radar.time)))])
    radar_plotfield[:,len(radar.time):]=radar.transpose("height", "time").values
    fig = plt.figure(figsize=(14,8))
    grid = plt.GridSpec(2, 2, width_ratios=[4,1])#Define grid and specify over how many cells axes spread
    ax1 = fig.add_subplot(grid[0,0])
    ax2 = fig.add_subplot(grid[1,0])
    ax3 = fig.add_subplot(grid[0,1])
    if w is not None:
        ax4 = fig.add_subplot(grid[1,1])
        ax4.plot(w,w.height)
        ax4.set_xlabel("Vertical Wind")
        ax4.set_ylabel("Height")
        ax4.grid(True)
    im1=ax1.imshow(cloudfield.isel(time=0).transpose("height", "pos").values, origin='lower', extent=(cloudfield.pos.min(), cloudfield.pos.max(), cloudfield.height.min(), cloudfield.height.max()), aspect=cloudaspect, **imargs)
    ax1.axvline(0, color='red')
    ax1.set_xlabel("Distance to Radar")
    ax1.set_ylabel("Height")
    im2=ax2.imshow(radar_plotfield.transpose("height", "reltime").values, origin='lower', extent=(radar_plotfield.reltime[0], radar_plotfield.reltime[-1], radar_plotfield.height.min(), radar_plotfield.height.max()), **imargs)
    ax2.axvline(0, color='red')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Height")
    ax3.plot(v, v.height)
    ax3.set_xlabel("Horizontal Wind")
    ax3.set_ylabel("Height")
    ax3.grid(True)

    def animate(i):
        im1.set_array(cloudfield.isel(time=i).transpose("height", "pos").values)
        im2.set_array(radar_plotfield.roll(reltime=-i).transpose("height", "reltime").values)
        return [im1, im2]

    anitimes=range(len(cloudfield.time))
    if len(anitimes)>50:
        anitimes=np.linspace(0,len(anitimes)-1, 50).astype(int)
    ani=FuncAnimation(fig, animate, anitimes, blit=True)
    return ani
#%%


#Locomotive
# filepath=workdir/'Locomotive.ppm'
# radar=radar_from_image(filepath)
# v=xr.DataArray(np.linspace(0.5,3,len(radar.height)), coords=[radar.height])
# cloudfield=retrieve_cloudfield(radar, v, 200)
# ani=make_animation(radar, cloudfield, v)
# ani.save(workdir/'Locomotive.mp4', progress_callback =lambda i, n: print(f'Saving frame {i} of {n}'))

#Box
# radar=xr.DataArray(np.ones((30,30)),coords=[('height', np.arange(30)), ('time', np.arange(30))] )
# v=xr.DataArray(np.linspace(0.5,3,len(radar.height)), coords=[radar.height])
# cloudfield=retrieve_cloudfield(radar, v, 200)
# cloudfield=cloudfield.fillna(0.0)
# ani=make_animation(radar, cloudfield, v)
# ani.save(workdir/'Box.mp4', progress_callback =lambda i, n: print(f'Saving frame {i} of {n}'))

#Ellipse
# filepath=workdir/'Ellipse.ppm'
# radar=radar_from_image(filepath)
# v=xr.DataArray(np.linspace(0.5,3,len(radar.height)), coords=[radar.height])
# cloudfield=retrieve_cloudfield(radar, v, 200)
# cloudfield=cloudfield.fillna(0.0)
# ani=make_animation(radar, cloudfield, v)
# ani.save(workdir/'Ellipse.mp4', progress_callback =lambda i, n: print(f'Saving frame {i} of {n}'))

#Ellipse with exotic wind profile
# filepath=workdir/'Ellipse.ppm'
# radar=radar_from_image(filepath)
# v=xr.DataArray(gaussian(len(radar.height), 50), coords=[radar.height])
# cloudfield=retrieve_cloudfield(radar, v, 200)
# cloudfield=cloudfield.fillna(0.0)
# ani=make_animation(radar, cloudfield, v)
# ani.save(workdir/'Ellipse_gauss.mp4', progress_callback =lambda i, n: print(f'Saving frame {i} of {n}'))
# fig, ax=plt.subplots()
# ax.plot(v, v.height)
# ax.grid(True)
# ax.set_xlabel("Windspeed")
# ax.set_ylabel("Height")
# fig.savefig(workdir/ "Gaussian_wind.jpg")

#Riming case 2018
# ds=xr.open_dataset(workdir/"20180312_event.nc")
# radar=ds.radar
# radar["time"]=((radar.time-radar.time[0])/1e9).astype(int)
# v=ds.v
# Pl, Pr=get_extent(radar.notnull(), v)
# cloudfield=retrieve_cloudfield(radar, v, 200)
#Riming case 2021-12-10
data_mahn=xr.open_dataset(workdir/"20211210_mahnwindfield.nc")
cloudfields=[]
for s in data_mahn.swipe_time:
    ds=data_mahn.sel(swipe_time=s)
    radar=ds.radar
    v=ds.vh
    cloudfields.append(retrieve_cloudfield(radar, v, 200, plimit=[-25000, 25000]))
cloudfields=xr.concat(cloudfields, 'swipe_time')
data_mahn["cloudfield"]=cloudfields
#%%
# anitime=np.datetime64("2021-12-10 09:45:00")
# anitime=np.datetime64("2021-12-10 10:00:00")
anitime=np.datetime64("2021-12-10 10:15:00")
ds_ani=data_mahn.sel(swipe_time=anitime, method='nearest')
ani=make_animation(ds_ani.radar, ds_ani.cloudfield, ds_ani.v, cloudaspect=10, imargs={"vmin":-3, "vmax":0, "cmap":'jet'})
ani.save(anidir/f'{ds_ani.swipe_time.dt.strftime("%Y%m%d_%H%M%S").item()}_mahnwindfield.mp4', progress_callback =lambda i, n: print(f'Saving frame {i} of {n}'))
#%%
data_mahn["advected"]=cloudfields
data_mahn.to_netcdf(outdir/"20211210_mahnwindfield.nc")
#Test extent plot
#%%
# fig, ax=plt.subplots()
# plottimeindex=0
# cloudfield.isel(time=plottimeindex).plot(x='pos',ax=ax)
# Pl.isel(time=plottimeindex).plot(y='height')
# Pr.isel(time=plottimeindex).plot(y='height')
# extent=(Pr.max('height')-Pl.min('height'))

# cloudfield=cloudfield.fillna(0.0)
# #%%
# minindex=(cloudfield>0).argmax('pos')
# minindex=minindex.where(minindex>0, drop=True).astype(int)
# minpos=cloudfield.pos[minindex].min('height')

# maxindex=(cloudfield>0).cumsum('pos').argmax('pos')
# maxindex=maxindex.where(maxindex>0, drop=True).astype(int)
# maxpos=cloudfield.pos[maxindex].max('height')
# spread=maxpos-minpos
# #%%
# ani=make_animation(radar, cloudfield, v, cloudaspect=15)
# ani.save(workdir/'Riming_20180315.mp4', progress_callback =lambda i, n: print(f'Saving frame {i} of {n}'))

#Box with fallspeed
# radar=xr.DataArray(np.ones((30,30)),coords=[('height', np.arange(30)), ('time', np.arange(30))] )
# radar[:,15:17]=0.0
# radar[15:17,:]=0.0
# v_height=np.arange(50)
# v=xr.DataArray(np.linspace(0.5,3,len(v_height)), coords=[("height",v_height)])
# w=0*v-0.5
# cloudfield=get_cloudfield(radar, v,w, 200)
# # cloudfield=cloudfield.fillna(0.0)
# ani=make_animation(radar, cloudfield, v,w)
# ani.save(workdir/'FallingBox.mp4', progress_callback =lambda i, n: print(f'Saving frame {i} of {n}'))

#Falling rain
# ds=xr.open_dataset(workdir/"20220330_rain.nc")
# radar=ds.radar
# radar["time"]=((radar.time-radar.time[0])/1e9).astype(float)
# radar["height"]=radar.height.astype(float)
# radar=radar.astype(float)
# radar=radar.fillna(0.0)
# v=ds.v
# w=ds.w
# # time_at_radar, height_at_radar=get_cloudfield(radar,v,w)
# cloudfield=get_cloudfield(radar, v,w, 200)
# cloudfield=cloudfield.sel(height=slice(1500,8000))
# # cloudfield=cloudfield.fillna(0.0)
# ani=make_animation(radar, cloudfield, v,w, cloudaspect=5)
# ani.save(workdir/'Rain_20220330.mp4', progress_callback =lambda i, n: print(f'Saving frame {i} of {n}'))




#%%
#linear profile
# v=xr.DataArray(np.linspace(0.5,3,len(radar.height)), coords=[radar.height])
# step profile
# v=xr.DataArray(np.ones(len(radar.height)), coords=[radar.height])
# v[int(len(v.height)/2):]=-1
# sine profile
# v=xr.DataArray(np.sin(np.linspace(0,2*np.pi, len(radar.height))), coords=[radar.height])
# cloudfield=retrieve_cloudfield(radar, v, 200)
# %%


sys.exit()
#%%

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('lightgrey')
mesh.set_facecolor('blue')
mesh.set_alpha(0.3)
ax.add_collection3d(mesh)
def plot_vertical_line(x,y,ax):
    ax.plot([x,x], [y,y], [0,250], color='red')

def plot_xz_plane(y, ax):
    xx, yy = np.meshgrid(range(150), range(150))
    xx=np.array([[0,100], [0,100]])
    yy=np.array([[y,y], [y,y]])
    z=np.array([[200,200], [0,0]])
    # z =0* xx+50#(9 - xx - yy) / 2 
    ax.plot_surface(xx,yy,z, color='red', alpha=0.5)

# plot_vertical_line(100,50,ax)
# plot_xz_plane(20, ax)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_xlim(verts[:,0].min(), verts[:,0].max())  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(verts[:,1].min(), verts[:,1].max())  # a = 6 (times two for 2nd ellipsoid)
ax.set_zlim(verts[:,2].min(), verts[:,2].max())  # a = 6 (times two for 2nd ellipsoid)

plt.tight_layout()
plt.show()

