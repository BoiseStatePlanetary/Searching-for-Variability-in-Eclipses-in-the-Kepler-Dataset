# 2018 Aug 9 - This file contains routines tailored to our analysis of 
#   eclipse variability.

import numpy as np
from glob import glob
from lightkurve import KeplerLightCurveFile
from astropy import units as u
from transit_utils import median_boxcar_filter, flag_outliers, transit_duration, fit_eclipse_bottom
from evilmc import evparams, evmodel, convert_Kz

class Kepler76_params:
    KIC = 4570949

    # The system parameters as reported in Faigler et al. (2013) --
    #  http://iopscience.iop.org/article/10.1088/0004-637X/771/1/26/meta
    a = 1./0.221 #\pm 0.003
    b = 0.944 # \pm 0.011
    inc = 78.0 # \pm 0.2
    per = 1.54492875*u.day # \pm 0.00000027
    Kz_m_per_s = 308. # \pm 20.
    Kz = convert_Kz(Kz=Kz_m_per_s) # convert to fraction of the speed of light

    Mp = 2.00*u.jupiterMass # \pm 0.26
    Rp_over_a = 0.0214 # \pm 0.0008
    Rp_over_Rs = Rp_over_a*a
    Ms = 1.2*u.solMass # \pm 0.2
    q = (Mp.to('kg')/Ms.to('kg')).value

    vsini = 6500.*u.m/u.s # \pm 2000
    Rs = 1.32*u.solRad #\pm 0.08
    Omegas = vsini.to('m/s')/Rs.to('m')*per.to('s') # stellar rotation state - very little effect, so chosen arbitrarily

    Ts = 6300. # \pm 200
    Faigler_T0 = (737.49 + 2455000. - 2454833.)# % per.to('day').value # \pm0.19
    T0 = 0.68508434

    coeffs = [0.313, 0.304]

    Aplanet = 60.4e-6 # \pm 2.0
    F0 = 60.4e-6 # \pm 2.0 -- overall shift in light curve, which is arbitrary
    phase_shift = -10.3/360. # \pm 2.0 - convert phase shift angle from degrees to orbital phase

    beta = 0.0716671 # Interpolation from among the values reported in Claret & Bloemen (2011) A&A 529, 75

    # Save parameters to an evilmc parameters object
    saved_ep = evparams(per=per.to('day').value, a=a, T0=T0, p=Rp_over_Rs, 
            limb_dark="quadratic", b=b, 
            F0=F0, Aplanet=Aplanet, phase_shift=phase_shift, beta=beta, 
            q=q, Kz=Kz, Ts=Ts, Ws=[0.,0.,Omegas], u=coeffs)

    saved_params = {
            "per": saved_ep.per,
            "i": inc,
            "a": saved_ep.a,
            "T0": saved_ep.T0,
            "p": saved_ep.p,
            "linLimb": saved_ep.u[0],
            "quadLimb": saved_ep.u[1],
            "b": saved_ep.b,
            "Aellip": 21.1e-6,
            "Abeam": 13.5e-6,
            "F0": 0.,
            "Aplanet": 50.4e-6,
            "phase_shift": saved_ep.phase_shift
            }

    # Estimate scatter
    dur = transit_duration(saved_ep)

def transit_indices(time, dur, T0):
    # Return transit indices
    return (np.abs(time - T0) < 0.5*dur)

def calc_SR(time, flux, err):
    # Calculate the signal residue as defined in Kovacs+ (2002) -
    # http://adsabs.harvard.edu/abs/2002A%26A...391..369K.
    # for the eclipse
    w = err**(-2)/np.sum(1./err**2.)   
    x = flux - np.mean(w*flux)/w

    s = np.sum(w*x)
    r = np.sum(w)
    
    return np.sqrt(s**2./(r*(1. - r)))

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

        # Remove nans since remove_nans above doesn't seem to work.
        ind = ~np.isnan(cur_flux)
        cur_time = cur_time[ind]
        cur_flux = cur_flux[ind]

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
