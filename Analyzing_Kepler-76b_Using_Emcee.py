# 2018 Oct 7 - Transferring the code in the ipynb to a script

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u

from statsmodels.robust.scale import mad
from scipy.optimize import curve_fit
from numpy.random import normal

from lightkurve import KeplerLightCurveFile

from evilmc import evparams, evmodel, convert_Kz
from transit_utils import bindata, transit_duration
from variable_eclipses_utils import *

from BEER_curve import BEER_curve

import emcee
from emcee.autocorr import integrated_time

import dill

K76 = Kepler76_params()

bounds = ([0., -1, K76.T0*0.95, -500e-6, -500e-6, -500e-6, 0., -1], 
                  [0.2, 1., K76.T0*1.05, 500e-6, 500e-6, 500e-6, 500e-6, 1])

def lnprior(theta):
    accept = True
    for i in range(len(theta)):
        accept = accept & (theta[i] > bounds[0][i] < theta[i] < bounds[1][i])

        if(accept):
            return 0.
        else:
            return -np.inf

def lnlike(theta, time, data, err):
    model = fit_all_signals(time, *theta)
    inv_sigma2 = 1.0/(err**2)

    return -0.5*(np.sum((data - model)**2*inv_sigma2)) 

def lnprob(theta, time, data, err):
    lp = lnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, time, data, err)

def fit_all_signals(cur_time, cur_p, cur_b, cur_T0, cur_Aellip, cur_Abeam,
        cur_F0, cur_eclipse_depth, cur_phase_shift): 
    params = K76.saved_params.copy()
            
    params['p'] = cur_p
    params['b'] = cur_b
    params['T0'] = cur_T0
    params['Aellip'] = cur_Aellip
    params['Abeam'] = cur_Abeam
    params['F0'] = cur_F0
    params['Aplanet'] = cur_eclipse_depth - cur_F0
    params['phase_shift'] = cur_phase_shift
                                                
    cur_BC = BEER_curve(cur_time, params, 
            supersample_factor=10, exp_time=30./60./24.)
                                                        
    return cur_BC.all_signals()

num_period = 1
binsize = 30./60./24.
    
# Retrieve while masking out transit
time, flux, filtered_time, filtered_flux = retreive_data(K76.saved_ep.per, 
        num_periods=num_period, KIC=K76.KIC, fit_bottom=True, 
        params=K76.saved_ep, drop_outliers=True)
folded_time = filtered_time % K76.saved_ep.per

# Estimate scatter
ind = ~transit_indices(folded_time, 2.*K76.dur, K76.T0)
unbinned_noise = mad(filtered_flux[ind])

# Estimate scatter
ind = ~transit_indices(folded_time, K76.dur, K76.T0)
unbinned_noise = mad(filtered_flux[ind])

time = folded_time
data = filtered_flux
err = unbinned_noise*np.ones_like(folded_time)

initial_guess = [K76.saved_params['p'], K76.saved_params['b'], 
        K76.saved_params['T0'], K76.saved_params['Aellip'], 
        K76.saved_params['Abeam'], K76.saved_params['F0'], 
        K76.saved_params['F0'] + K76.saved_params['Aplanet'], 
        K76.saved_params['phase_shift']]
popt, pcov = curve_fit(fit_all_signals, time, data, sigma=err, 
        p0=initial_guess, bounds=bounds)
print(popt)
uncertainty = np.sqrt(pcov.diagonal())

ndim, nwalkers = len(initial_guess), 24
pos = [popt + uncertainty*np.random.randn(ndim) for i in range(nwalkers)]

print("Running sampler")
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, data, err))

filename = 'Analyzing_Kepler-76b_Using_Emcee.pkl'
nsteps = 5000
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    if (i+1) % 50 == 0:
        print("{0:5.1%}".format(float(i) / nsteps))
        # Incrementally save progress
#       dill.dump_session(filename)

dill.dump_session(filename)
print(np.mean(sampler.chain[:, :, 0]), np.std(sampler.chain[:, :, 0]))

