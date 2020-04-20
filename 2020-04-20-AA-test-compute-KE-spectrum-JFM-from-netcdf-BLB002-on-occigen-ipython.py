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
import xscale.spectral.fft as xfft
import xscale 
import Wavenum_freq_spec_func as wfs
import time



# In[ ]:


get_ipython().magic(u'time')

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


get_ipython().run_cell_magic(u'time', u'', u'\nprint(\'Select dates\')\nu_JFM=dsu.sel(time_counter=slice(\'2010-01-01\',\'2010-03-31\'))[\'somecrty\']\nv_JFM=dsv.sel(time_counter=slice(\'2010-01-01\',\'2010-03-31\'))[\'somecrty\']\n\n\n\nprint(\'Select box area\')\nu_JFM_box=u_JFM[:,jmin:jmax,imin:imax].chunk({\'time_counter\':10,\'x\':120,\'y\':120})\nv_JFM_box=v_JFM[:,jmin:jmax,imin:imax].chunk({\'time_counter\':10,\'x\':120,\'y\':120})\n\n\n\n# - get dx and dy\nprint(\'get dx and dy\')\ndx_JFM,dy_JFM = wfs.get_dx_dy(u_JFM_box[0],lonbox,latbox)\n\n\n#... Detrend data in all dimension ...\nprint(\'Detrend data in all dimension\')\nu_JFM = wfs.detrendn(u_JFM_box,axes=[0,1,2])\nv_JFM = wfs.detrendn(v_JFM_box,axes=[0,1,2])\n\n#... Apply hanning windowing ...\') \nprint(\'Apply hanning windowing\')\nu_JFM = wfs.apply_window(u_JFM, u_JFM.dims, window_type=\'hanning\')\nv_JFM = wfs.apply_window(v_JFM, v_JFM.dims, window_type=\'hanning\')\n\n\n#... Apply hanning windowing ...\') \nprint(\'FFT \')\nu_JFMhat = xfft.fft(u_JFM, dim=(\'time_counter\', \'x\', \'y\'), dx={\'x\': dx_JFM, \'y\': dx_JFM}, sym=True)\nv_JFMhat = xfft.fft(v_JFM, dim=(\'time_counter\', \'x\', \'y\'), dx={\'x\': dx_JFM, \'y\': dx_JFM}, sym=True)\n\n#... Apply hanning windowing ...\') \nprint(\'PSD \')\nu_JFM_psd = xfft.psd(u_JFMhat)\nv_JFM_psd = xfft.psd(v_JFMhat)\n\n\n#... Get frequency and wavenumber ... \nprint(\'Get frequency and wavenumber\')\nfrequency_JFM = u_JFMhat.f_time_counter\nkx_JFM = u_JFMhat.f_x\nky_JFM = u_JFMhat.f_y\n\n#... Get istropic wavenumber ... \nprint(\'Get istropic wavenumber\')\nwavenumber_JFM,kradial_JFM = wfs.get_wavnum_kradial(kx_JFM,ky_JFM)\n\n#... Get numpy array ... \nprint(\'Get numpy array\')\nu_JFM_psd_np = u_JFM_psd.values\nv_JFM_psd_np = v_JFM_psd.values\n\n#... Get 2D frequency-wavenumber field ... \nprint(\'Get f k in 2D\')\nu_JFM_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,u_JFM_psd_np)\nv_JFM_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,v_JFM_psd_np)\n\nKE_JFM_wavenum_freq_spectrum=0.5*(u_JFM_wavenum_freq_spectrum+v_JFM_wavenum_freq_spectrum)\n\n# Save to Netscdf file\n# - build dataarray\nprint(\'Save to Netscdf file\')\nKE_JFM_wavenum_freq_spectrum_da = xr.DataArray(KE_JFM_wavenum_freq_spectrum,dims=[\'frequency\',\'wavenumber\'],name="Ke_spectrum",coords=[frequency_JFM ,wavenumber_JFM])\nKE_JFM_wavenum_freq_spectrum_da.attrs[\'Name\'] = \'KE_Spectrum_JFM_w_k_from_1h_eNATL60-BLB002\'\n\nKE_JFM_wavenum_freq_spectrum_da.to_dataset().to_netcdf(path=\'test2-KE_Spectrum_JFM_w_k_from_1h_eNATL60-BLB002.nc\',mode=\'w\',engine=\'scipy\')')

