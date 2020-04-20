#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python


import xarray as xr
import dask
import dask.threaded
import dask.multiprocessing
from dask.distributed import Client
import numpy as np                                                                                        
import zarr



# In[ ]:


c = Client()

c


# In[ ]:


import sys, glob
import numpy as np
import xarray as xr
sys.path.insert(0,"/scratch/cnt0024/hmg2840/albert7a/DEV/git/xscale")
import xscale.spectral.fft as xfft
import xscale 
import Wavenum_freq_spec_func as wfs
import time



# In[ ]:


dir_data='/scratch/cnt0024/hmg2840/albert7a/eNATL60/eNATL60-BLB002-S/1h/surf/'
dsu=xr.open_mfdataset(dir_data+'*sozocrtx*',combine='by_coords',parallel=True,chunks={'time_counter':1,'x':1000,'y':1000})
dsv=xr.open_mfdataset(dir_data+'*somecrty*',combine='by_coords',parallel=True,chunks={'time_counter':1,'x':1000,'y':1000})




# In[ ]:



lat=dsv['nav_lat']
lon=dsv['nav_lon']
 
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


# In[ ]:


print('Select dates')
u_JFM=dsu.sel(time_counter=slice('2010-01-01','2010-03-31'))['somecrty']
v_JFM=dsv.sel(time_counter=slice('2010-01-01','2010-03-31'))['somecrty']



print('Select box area')
u_JFM_box=u_JFM[:,jmin:jmax,imin:imax].chunk({'time_counter':10,'x':120,'y':120})
v_JFM_box=v_JFM[:,jmin:jmax,imin:imax].chunk({'time_counter':10,'x':120,'y':120})



# - get dx and dy
print('get dx and dy')
dx_JFM,dy_JFM = wfs.get_dx_dy(u_JFM_box[0],lonbox,latbox)


#... Detrend data in all dimension ...
print('Detrend data in all dimension')
u_JFM = wfs.detrendn(u_JFM_box,axes=[0,1,2])
v_JFM = wfs.detrendn(v_JFM_box,axes=[0,1,2])

#... Apply hanning windowing ...') 
print('Apply hanning windowing')
u_JFM = wfs.apply_window(u_JFM, u_JFM.dims, window_type='hanning')
v_JFM = wfs.apply_window(v_JFM, v_JFM.dims, window_type='hanning')


#... Apply hanning windowing ...') 
print('FFT ')
u_JFMhat = xfft.fft(u_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)
v_JFMhat = xfft.fft(v_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)

#... Apply hanning windowing ...') 
print('PSD ')
u_JFM_psd = xfft.psd(u_JFMhat)
v_JFM_psd = xfft.psd(v_JFMhat)


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
v_JFM_psd_np = v_JFM_psd.values

#... Get 2D frequency-wavenumber field ... 
print('Get f k in 2D')
u_JFM_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,u_JFM_psd_np)
v_JFM_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,v_JFM_psd_np)

KE_JFM_wavenum_freq_spectrum=0.5*(u_JFM_wavenum_freq_spectrum+v_JFM_wavenum_freq_spectrum)

# Save to Netscdf file
# - build dataarray
print('Save to Netscdf file')
KE_JFM_wavenum_freq_spectrum_da = xr.DataArray(KE_JFM_wavenum_freq_spectrum,dims=['frequency','wavenumber'],name="Ke_spectrum",coords=[frequency_JFM ,wavenumber_JFM])
KE_JFM_wavenum_freq_spectrum_da.attrs['Name'] = 'KE_Spectrum_JFM_w_k_from_1h_eNATL60-BLB002'

KE_JFM_wavenum_freq_spectrum_da.to_dataset().to_netcdf(path='test2-KE_Spectrum_JFM_w_k_from_1h_eNATL60-BLB002.nc',mode='w',engine='scipy')

