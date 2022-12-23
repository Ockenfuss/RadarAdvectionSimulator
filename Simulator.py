#%%
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
workdir=Path("/project/meteo/work/Paul.Ockenfuss/PhD/Experiments/RadarAdvectionSimulator/")


#%%
def retrieve_cloudfield(radar,v, sn=100):
    vmax=abs(v).max()
    deltat=radar.time.max()-radar.time.min()
    dist=xr.DataArray(np.linspace(-vmax*deltat, vmax*deltat, sn), dims=["dist"])
    dist.coords["dist"]=dist
    time_to_radar=-dist/v+radar.time
    height=radar.height.expand_dims({'time':radar.time, 'dist':dist})
    cloudfield=radar.interp(time=time_to_radar, height=height)
    return cloudfield

def radar_from_image(filepath):
    image=imread(filepath)
    image=image[::-1,:,3]
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


filepath=workdir/'Lines.png'
radar=radar_from_image(filepath)
#linear profile
v=xr.DataArray(np.linspace(1,5,len(radar.height)), coords=[radar.height])
# step profile
# v=xr.DataArray(np.ones(len(radar.height)), coords=[radar.height])
# v[int(len(v.height)/2):]=-1
# sine profile
# v=xr.DataArray(np.sin(np.linspace(0,2*np.pi, len(radar.height))), coords=[radar.height])
cloudfield=retrieve_cloudfield(radar, v)
#%%
verts, faces, normals, values = measure.marching_cubes(cloudfield.fillna(0.0).transpose("time", "dist", "height").values, 0, step_size=2)
# %%
deltat=radar.time.max()-radar.time.min()
radar_plotfield=xr.DataArray(np.zeros((len(radar.height), len(radar.time)*2)), coords=[("height", radar.height.values), ("reltime", np.linspace(deltat, -deltat, 2*len(radar.time)))])
radar_plotfield[:,len(radar.time):]=radar.values
fig = plt.figure(figsize=(14,8))
grid = plt.GridSpec(2, 2, width_ratios=[4,1])#Define grid and specify over how many cells axes spread
ax1 = fig.add_subplot(grid[0,0])
ax2 = fig.add_subplot(grid[1,0])
ax3 = fig.add_subplot(grid[0,1])
ax4 = fig.add_subplot(grid[1,1], projection='3d')
im1=ax1.imshow(cloudfield.isel(time=0).transpose("height", "dist").values, origin='lower', extent=(cloudfield.dist.min(), cloudfield.dist.max(), cloudfield.height.min(), cloudfield.height.max()))
ax1.axvline(0, color='red')
ax1.set_xlabel("Distance")
ax1.set_ylabel("Height")
im2=ax2.imshow(radar_plotfield.transpose("height", "reltime").values, origin='lower', extent=(radar_plotfield.reltime[0], radar_plotfield.reltime[-1], radar_plotfield.height.min(), radar_plotfield.height.max()))
ax2.axvline(0, color='red')
ax2.set_xlabel("Time")
ax2.set_ylabel("Height")
ax3.plot(v, v.height)
ax3.set_xlabel("V Wind")
ax3.set_ylabel("Height")
ax3.grid(True)

#3D Plot
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('lightgrey')
# mesh.set_facecolor('blue')
# mesh.set_alpha(0.3)
ax4.add_collection3d(mesh)

# plot_vertical_line(100,50,ax4)
# plot_xz_plane(20, ax4)

ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_zlabel("z")
# ax4.set_xticklabels([])

ax4.set_xlim(verts[:,0].min(), verts[:,0].max())  # a = 6 (times two for 2nd ellipsoid)
ax4.set_ylim(verts[:,1].min(), verts[:,1].max())  # a = 6 (times two for 2nd ellipsoid)
ax4.set_zlim(verts[:,2].min(), verts[:,2].max())  # a = 6 (times two for 2nd ellipsoid)
#%%
def animate(time):
    im1.set_array(cloudfield.sel(time=time).transpose("height", "dist").values)
    global radar_plotfield
    im2.set_array(radar_plotfield.transpose("height", "reltime").values)
    radar_plotfield=radar_plotfield.roll(reltime=-1)
    return [im1, im2]
ani=FuncAnimation(fig, animate, cloudfield.time, blit=True)
ani.save(workdir/'Animation.mp4')

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

