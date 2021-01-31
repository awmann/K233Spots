#GAUSSFOLD.py
#Keegan Marr
#Last Modified: March 18, 2019

#Convolves an input function with a Gaussian profile

import numpy as np
from scipy import interpolate
import time
import matplotlib.pyplot as plt


#Makes a 1D Gaussian defined by its FWHM
def gaussian1d(npix, fwhm, normalize=True):
    # Initialize Gaussian params 
    cntrd = (npix + 1.0) * 0.5 
    st_dev = 0.5 * fwhm / np.sqrt( 2.0 * np.log(2) )
    x = np.linspace(1,npix,npix)
    
    # Make Gaussian
    ampl = (1/(np.sqrt(2*np.pi*(st_dev**2))))
    expo = np.exp(-((x - cntrd)**2)/(2*(st_dev**2)))
    gaussian = ampl * expo
    
    # Normalize
    if normalize:  gaussian /= gaussian.sum() 
    
    return gaussian
    

def gaussfold(lam, flux, fwhm):

    lammin = min(lam)
    lammax = max(lam)

    dlambda = fwhm / float(17)

    interlam = lammin + dlambda * np.arange(float((lammax-lammin)/dlambda+1))
    x = interpolate.interp1d(lam, flux, kind='linear', fill_value='extrapolate')
    interflux = x(interlam)

    fwhm_pix = fwhm / dlambda
    window = int(17 * fwhm_pix)

    # Get a 1D Gaussian Profile
    gauss = gaussian1d(window, fwhm_pix)
    
    # Convolve input spectrum with the Gaussian profile
    fold = np.convolve(interflux, gauss, mode='same')

    y = interpolate.interp1d(interlam, fold, kind='linear', fill_value='extrapolate')
    fluxfold = y(lam)

    return fluxfold




