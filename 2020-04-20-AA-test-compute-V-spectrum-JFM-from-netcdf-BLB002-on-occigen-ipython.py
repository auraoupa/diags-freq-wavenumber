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




dir_data='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLB002-S/1h/surf/'
dsv1=xr.open_mfdataset(dir_data+'*m01**somecrty*',combine='by_coords',parallel=True,chunks={'time_counter':1,'x':1000,'y':1000})
dsv2=xr.open_mfdataset(dir_data+'*m02**somecrty*',combine='by_coords',parallel=True,chunks={'time_counter':1,'x':1000,'y':1000})
dsv3=xr.open_mfdataset(dir_data+'*m03**somecrty*',combine='by_coords',parallel=True,chunks={'time_counter':1,'x':1000,'y':1000})





lat=dsv1['nav_lat']
lon=dsv1['nav_lon']
 
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
v_1=dsv1['somecrty']
v_2=dsv2['somecrty']
v_3=dsv3['somecrty']
v_JFM=xr.concat([v_1,v_2,v_3],dim='time_counter')



print('Select box area')
v_JFM_box=v_JFM[:,jmin:jmax,imin:imax].chunk({'time_counter':10,'x':120,'y':120})



# - get dx and dy
print('get dx and dy')
dx_JFM,dy_JFM = wfs.get_dx_dy(v_JFM_box[0],lonbox,latbox)


#... Detrend data in all dimension ...
print('Detrend data in all dimension')
v_JFM = wfs.detrendn(v_JFM_box,axes=[0,1,2])

#... Apply hanning windowing ...') 
print('Apply hanning windowing')
v_JFM = wfs.apply_window(v_JFM, v_JFM.dims, window_type='hanning')


#... Apply hanning windowing ...') 
print('FFT ')
v_JFMhat = xfft.fft(v_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)

#... Apply hanning windowing ...') 
print('PSD ')
v_JFM_psd = xfft.psd(v_JFMhat)


#... Get frequency and wavenumber ... 
print('Get frequency and wavenumber')
frequency_JFM = v_JFMhat.f_time_counter
kx_JFM = v_JFMhat.f_x
ky_JFM = v_JFMhat.f_y

#... Get istropic wavenumber ... 
print('Get istropic wavenumber')
wavenumber_JFM,kradial_JFM = wfs.get_wavnum_kradial(kx_JFM,ky_JFM)

#... Get numpy array ... 
print('Get numpy array')
v_JFM_psd_np = v_JFM_psd.values

#... Get 2D frequency-wavenumber field ... 
print('Get f k in 2D')
v_JFM_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,v_JFM_psd_np)

# Save to Netscdf file
# - build dataarray
print('Save to Netscdf file')
V_JFM_wavenum_freq_spectrum_da = xr.DataArray(V_JFM_wavenum_freq_spectrum,dims=['frequency','wavenumber'],name="V_spectrum",coords=[frequency_JFM ,wavenumber_JFM])
V_JFM_wavenum_freq_spectrum_da.attrs['Name'] = 'V_Spectrum_JFM_w_k_from_1h_eNATL60-BLB002'

V_JFM_wavenum_freq_spectrum_da.to_dataset().to_netcdf(path='test-netcdf-V_Spectrum_JFM_w_k_from_1h_eNATL60-BLB002.nc',mode='w',engine='scipy')

