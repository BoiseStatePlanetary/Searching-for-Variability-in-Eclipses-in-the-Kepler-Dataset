# 2018 Aug 9 - This file contains routines tailored to our analysis of 
#   eclipse variability.

import numpy as np
from glob import glob
from lightkurve import KeplerLightCurveFile, KeplerTargetPixelFile
from astropy import units as u
from transit_utils import median_boxcar_filter, flag_outliers, transit_duration, fit_eclipse_bottom, supersample_time
from evilmc import evparams, evmodel, convert_Kz
from PyAstronomy.modelSuite.XTran.forTrans import MandelAgolLC

class Kepler76_params:
    KIC = 4570949

    # exposure time for Kepler observations
    exp_time = 30./60./24.
    supersample_factor = 10

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

    coeffs = [0.313, 0.304] # From Faigler et al. (2013)

    Aplanet = 60.4e-6 # \pm 2.0
    F0 = 60.4e-6 # \pm 2.0 -- overall shift in light curve, which is arbitrary
    phase_shift = -10.3/360. # \pm 2.0 - convert phase shift angle from degrees to orbital phase
    Aellip = 21.1e-6
    Abeam = 13.5e-6
    eclipse_depth = 98.9e-6

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

def redchisqg(ydata,ymod,deg,sd):
    """
    Returns the reduced chi-square error statistic for an arbitrary model,
    chisq/nu, where nu is the number of degrees of freedom. If individual
    standard deviations (array sd) are supplied, then the chi-square error
    statistic is computed as the sum of squared errors divided by the standard
    deviations. See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.

    ydata,ymod,sd assumed to be Numpy arrays. deg integer.

    Usage:
    >>> chisq=redchisqg(ydata,ymod,n,sd)
    where
    ydata : data
    ymod : model evaluated at the same x points as ydata
    n : number of free parameters in the model
    sd : uncertainties in ydata

    Rodrigo Nemmen
    http://goo.gl/8S1Oo
    """
    # Chi-square statistic
    chisq=np.sum( ((ydata-ymod)/sd)**2 )

    # Number of degrees of freedom assuming 2 free parameters
    nu=ydata.size-1-deg

    return chisq/nu

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
        downloaded=True, base_dir="../mastDownload/Kepler/",
        params=None, fit_bottom=False, dilution=False):
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
        dilution (optional, boolean) - whether to calculate and multiply by
            by dilution factor; almost definitely you shouldn't use this!

    Returns:
        time (float array) - observational times
        flux (float array) - unconditioned light curve data
        filtered_time (float array) - observational times, conditioned
        filtered_flux (float array) - observational data, conditioned

    """

    if(not downloaded):
        for q in range(0, 18):
            try:
                lc = KeplerLightCurveFile.from_archive(str(KIC), 
                        quarter=q, verbose=False).PDCSAP_FLUX
                tpf = KeplerTargetPixelFile.from_archive(str(KIC), quarter=q)

            except:
                pass

    time = np.array([])
    flux = np.array([])

    filtered_time = np.array([])
    filtered_flux = np.array([])

    # Collect all data files
    ls = glob(base_dir + "kplr*" + str(KIC) + "_lc_Q*/*.fits")
    tpfs = glob(base_dir + "kplr*" + str(KIC) + "_lc_Q*/*targ.fits.gz")

    for i in range(len(ls)):
     # PDCSAP_FLUX supposedly takes care of the flux fraction - 
     # _Data Processing Handbook_, p. 129
     # https://archive.stsci.edu/kepler/manuals/KSCI-19081-002-KDPH.pdf
        lc = KeplerLightCurveFile(ls[i]).PDCSAP_FLUX
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
        # 2018 Dec 4 - Un-dilute the light curve (lightcurve?)
        crowdsap = KeplerTargetPixelFile(tpfs[i]).\
                header('TARGETTABLES')['CROWDSAP']
        dilution_factor = 2. - crowdsap

        if(dilution):
            cur_flux *= dilution_factor

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
            if(type(params) is dict):
                T0 = params['T0']
            else:
                T0 = params.T0
            ind = np.abs(folded_time - T0) < dur

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

def fit_transit(time, params, supersample_factor=10, exp_time=30./60./24.):
    time_supersample = supersample_time(time, supersample_factor, exp_time)
    
    ma = MandelAgolLC(orbit="circular", ld="quad")

    baseline = params["baseline"]

    ma["per"] = params["per"]
    ma["a"] = params["a"]
    ma["T0"] = params["T0"]
    ma["p"] = params["p"]
    ma["i"] = np.arccos(params["b"]/params["a"])*180./np.pi
    ma["linLimb"] = params["linLimb"]
    ma["quadLimb"] = params["quadLimb"]
    
    transit_supersample = ma.evaluate(time_supersample) - 1. + baseline
    return np.mean(transit_supersample.reshape(-1, supersample_factor), axis=1)

def stack_orbits(period, time, num_orbits=10, sliding_window=True, 
        max_time=None, min_orbits_to_fold=9, cadence_length=30./60./24.):
    """
    This routine will fold and stack orbits together, returning
    each stack of orbits.

    Args:
        period (float) - orbital period for the transiting planet
        time (numpy array) - all observational times
        num_orbits (optional, defaults to 10) - how many orbits to stack 
        sliding_window (optional, boolean) - whether the window slides or jumps
        max_time (optional, float) - what maximum time; If None, use max(time)
        min_orbits_to_fold (optional, int) - minimum number of orbits required
            to make new window, should probably be a little smaller than
            num_orbits
        cadence_length (optional, float) - how long each cadence in same units
            as time (defaults to 30 min in days)

    Returns:
        a dictionary for which the keys are the mid-orbit time and
          the values are the indices for each set of orbits
    """

    orbits = {}

    sliding_window_factor = 1.
    if(not sliding_window):
        sliding_window_factor = float(num_orbits*period)

    mn = np.min(time)
    mx = np.min(time) + num_orbits*period

    if(max_time is None):
        max_time = np.max(time)

    while(mx <= max_time - num_orbits*period):
        ind = ((time >= mn) & (time < mx))

        # Check that window spans required span
        num_cadences_in_window = float(time[ind].size)
        num_cadences_required = min_orbits_to_fold*period/cadence_length
        has_enough_orbits = num_cadences_in_window >= num_cadences_required

        mid_time = np.nanmedian(time[ind])
        if((has_enough_orbits) and (~np.isnan(mid_time))):
            orbits[mid_time] = ind

        mn += sliding_window_factor*period
        mx += sliding_window_factor*period

    return orbits
