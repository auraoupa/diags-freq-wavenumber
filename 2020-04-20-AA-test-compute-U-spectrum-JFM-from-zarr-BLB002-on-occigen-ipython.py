#!/usr/bin/env python


import xarray as xr
import dask
import dask.threaded
import dask.multiprocessing
from dask.distributed import Client
import numpy as np                                                                                        
import zarr



from dask_jobqueue import SLURMCluster 
from dask.distributed import Client 
  
cluster = SLURMCluster(cores=28,name='make_profiles',walltime='00:30:00',job_extra=['--constraint=HSW24','--exclusive','--nodes=1'],memory='120GB',interface='ib0') 
cluster.scale(196)
cluster

from dask.distributed import Client
client = Client(cluster)
client


import time
nb_workers = 0
while True:
    nb_workers = len(client.scheduler_info()["workers"])
    if nb_workers >= 2:
        break
    time.sleep(1)
print(nb_workers)



import sys, glob
import numpy as np
import xarray as xr
sys.path.insert(0,"/scratch/cnt0024/hmg2840/albert7a/DEV/git/xscale")

import xscale.spectral.fft as xfft
import xscale 
import Wavenum_freq_spec_func as wfs
import time




zarru1='/store/albert7a/eNATL60/zarr/zarr_eNATL60-BLB002-SSU-1h-y2010m01'
zarru2='/store/albert7a/eNATL60/zarr/zarr_eNATL60-BLB002-SSU-1h-y2010m02'
zarru3='/store/albert7a/eNATL60/zarr/zarr_eNATL60-BLB002-SSU-1h-y2010m03'
dsu1=xr.open_zarr(zarru1)
dsu2=xr.open_zarr(zarru2)
dsu3=xr.open_zarr(zarru3)




lat=dsu1['nav_lat']
lon=dsu1['nav_lon']
 
latmin = 40.0; latmax = 45.0;
lonmin = -40.0; lonmax = -35.0;

domain = (lonmin<lon) * (lon<lonmax) * (latmin<lat) * (lat<latmax)
where = np.where(domain)

#get indice
jmin = np.min(where[0][:])
jmax = np.max(where[0][:])
imin = np.min(where[1][:])
imax = np.max(where[1][:])

latbox=lat[jmin:jmax,imin:imax]
lonbox=lon[jmin:jmax,imin:imax]

print(jmin,jmax,imin,imax)



print('Select dates')
u_1=dsu1['sozocrtx']
u_2=dsu2['sozocrtx']
u_3=dsu3['sozocrtx']
u_JFM=xr.concat([u_1,u_2,u_3],dim='time_counter')



print('Select box area')
u_JFM_box=u_JFM[:,jmin:jmax,imin:imax].chunk({'time_counter':10,'x':120,'y':120})



# - get dx and dy
print('get dx and dy')
dx_JFM,dy_JFM = wfs.get_dx_dy(u_JFM_box[0],lonbox,latbox)


#... Detrend data in all dimension ...
print('Detrend data in all dimension')
u_JFM = wfs.detrendn(u_JFM_box,axes=[0,1,2])

#... Apply hanning windowing ...') 
print('Apply hanning windowing')
u_JFM = wfs.apply_window(u_JFM, u_JFM.dims, window_type='hanning')


#... Apply hanning windowing ...') 
print('FFT ')
u_JFMhat = xfft.fft(u_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)

#... Apply hanning windowing ...') 
print('PSD ')
u_JFM_psd = xfft.psd(u_JFMhat)


#... Get frequency and wavenumber ... 
print('Get frequency and wavenumber')
frequency_JFM = u_JFMhat.f_time_counter
kx_JFM = u_JFMhat.f_x
ky_JFM = u_JFMhat.f_y

#... Get istropic wavenumber ... 
print('Get istropic wavenumber')
wavenumber_JFM,kradial_JFM = wfs.get_wavnum_kradial(kx_JFM,ky_JFM)

#... Get numpy array ... 
print('Get numpy array')
u_JFM_psd_np = u_JFM_psd.values

#... Get 2D frequency-wavenumber field ... 
print('Get f k in 2D')
u_JFM_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,u_JFM_psd_np)

# Save to Netscdf file
# - build dataarray
print('Save to Netscdf file')
U_JFM_wavenum_freq_spectrum_da = xr.DataArray(U_JFM_wavenum_freq_spectrum,dims=['frequency','wavenumber'],name="U_spectrum",coords=[frequency_JFM ,wavenumber_JFM])
U_JFM_wavenum_freq_spectrum_da.attrs['Name'] = 'U_Spectrum_JFM_w_k_from_1h_eNATL60-BLB002'

U_JFM_wavenum_freq_spectrum_da.to_dataset().to_netcdf(path='test-netcdf-U_Spectrum_JFM_w_k_from_1h_eNATL60-BLB002.nc',mode='w',engine='scipy')

