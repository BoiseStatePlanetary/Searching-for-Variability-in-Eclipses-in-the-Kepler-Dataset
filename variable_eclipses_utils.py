# 2018 Aug 9 - This file contains routines tailored to our analysis of 
#   eclipse variability.

import numpy as np
from glob import glob
from lightkurve import KeplerLightCurveFile

from transit_utils import median_boxcar_filter, flag_outliers, transit_duration, fit_eclipse_bottom

def retreive_data(period, num_periods=2, KIC=4570949, drop_outliers=False, 
        downloaded=True, base_dir="mastDownload/Kepler/",
        params=None, fit_bottom=True):
    """
    Retreives and conditions data for the given KIC object

    Args:
        period (float) - orbital period for the transiting planet
        num_periods (int, optional) - window size for median filter
        KIC (optional, int) - KIC number
        drop_outliers (optional, boolean) - drop outliers?
        downloaded (optional, boolean) - whether data are DLed
        base_dir (optional, str) - directory under which to find data files
        params (optional, evilmc.evparams) - if not None, the routine masks
            points in transit (as indicated by the params values)
            while conditioning the data
        fit_bottom (optional, boolean) - whether to shift data and zero eclipse

    Returns:
        time (float array) - observational times
        flux (float array) - unconditioned light curve data
        filtered_time (float array) - observational times, conditioned
        filtered_flux (float array) - observational data, conditioned

    """

    if(not downloaded):
        for q in range(0, 18):
            lc = KeplerLightCurveFile.from_archive(str(KIC), 
                    quarter=q, verbose=False).PDCSAP_FLUX

    time = np.array([])
    flux = np.array([])

    filtered_time = np.array([])
    filtered_flux = np.array([])

    # Collect all data files
    ls = glob(base_dir + "kplr*" + str(KIC) + "_lc_Q*/*.fits")
    for cur_file in ls:
     # PDCSAP_FLUX supposedly takes care of the flux fraction - 
     # _Data Processing Handbook_, p. 129
     # https://archive.stsci.edu/kepler/manuals/KSCI-19081-002-KDPH.pdf
        lc = KeplerLightCurveFile(cur_file).PDCSAP_FLUX
        lc.remove_nans()

        cur_time = lc.time
        cur_flux = lc.flux

        time = np.append(time, cur_time)
        flux = np.append(flux, cur_flux)

        cur_flux = (cur_flux - np.nanmedian(cur_flux))/np.nanmedian(cur_flux)
        window = num_periods*period
        del_t = np.nanmedian(cur_time[1:] - cur_time[:-1])
        window_length = int(window/del_t)

        # Indicate all points in transit
        ind = None
        if(params is not None):
            folded_time = cur_time % period

            dur = transit_duration(params, which_duration='full')

            # This expression below should technically be
            # ind = np.abs(folded_time - params.T0) < 0.5*dur, but
            # I'm taking a little window to either side of the transit
            # to make sure I'm masking everything.
            ind = np.abs(folded_time - params.T0) < dur

        filt = median_boxcar_filter(cur_time, cur_flux, 
            window_length, mask_ind=ind)

        filtered_time = np.append(filtered_time, cur_time)
        filtered_flux = np.append(filtered_flux, cur_flux - filt)
    
    ind = ~np.isnan(filtered_flux)
    filtered_time = filtered_time[ind]
    filtered_flux = filtered_flux[ind]

    if(drop_outliers):
        ind = flag_outliers(filtered_flux)
        filtered_time = filtered_time[ind]
        filtered_flux = filtered_flux[ind]

    if(fit_bottom):
        eclipse_bottom = fit_eclipse_bottom(filtered_time, filtered_flux,
                params)
        filtered_flux -= eclipse_bottom

    # Finally remove any NaNs that snuck through
    ind = ~np.isnan(filtered_flux)
    filtered_time = filtered_time[ind]
    filtered_flux = filtered_flux[ind]

    return time, flux, filtered_time, filtered_flux
