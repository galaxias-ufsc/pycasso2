'''
Created on 08/24/2016

@author: ariel

Provides functions to correct spectra for galactic extinction
'''

import numpy as np
from os import listdir
import urllib
import healpy as hp
from . import wcs

def get_EBV_map(file_name, data_dir):
    '''
    
    Reads E(B-V) HEALPix map from Planck's dust map using healpy, if the map 
    file is not found, it will be downloaded from:
    http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_CompMap_ThermalDustModel_2048_R1.20.fits
    
    
    '''
    
    data_dir += '/'    
    
    if file_name not in listdir(data_dir):
        print 'Downloading dust map (1.5GB), this is probably a good time to check XKCD.'        
        
        url = 'http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_CompMap_ThermalDustModel_2048_R1.20.fits'
    
        urllib.urlretrieve(url, data_dir + file_name)
    
    print 'Reading E(B-V) map from ' + data_dir + file_name
    EBV_map = hp.read_map(data_dir + file_name, field = 2)
    
    return EBV_map
    
    
def CCM(lamb, Rv= 3.1):
    '''
    
    Calculates the Cardelli, Clayton & Mathis (CCM) extinction curve in the 
    optical, wavelengths should be in the 3030-9090 Angstrons range.
    
    Input:   Wavelengths, Rv (Optional, default is 3.1)
    Returns: A_lambda/Av
    
    Reference: http://adsabs.harvard.edu/abs/1989ApJ...345..245C
    
    '''
    
    #Turn lambda from angstrons to microns:
    lamb = lamb / 10000. 
    
    #Calculate A_lambda/Av:
    x =  1./lamb
    y =  (x-1.82)
    
    a =  1.+(0.17699 * y) - ( 0.50447 * (y **2) ) - ( 0.02427 * (y **3) )
    a += ( 0.72085 * (y **4) ) + (0.01979 * (y **5) ) - ( 0.77530 * (y **6) )
    a += (0.32999 * (y **7) )
    
    b =  (1.41338 * y) + ( 2.28305 * (y **2) ) + ( 1.07233 * (y **3) )
    b += -( 5.38434 * (y **4) ) - (0.62251 * (y **5) ) + ( 5.30260 * (y **6) )
    b += -(2.09002 * (y **7) )
    
    A_lambda_Av = a + (b/Rv)
    
    return A_lambda_Av
    
    
def calc_extinction(ra,dec,lamb,EBVmap,Rv=3.1):
    '''
    
    Gets the galactic extinction in a given wavelenght through a given line of 
    sight.
    
    Input:   RA, Dec, wavelenght, E(B-V) map, Rv (Optional, default is 3.1)
    Returns: A_lambda, E(B-V)
    
    '''
    
    #Turn RA, Dec into galactic coordinates:
    l, b = wcs.get_galactic_coordinates(ra,dec)   
    
    #Get the corresponting HEALPix index and the E(B-V) value:
    index = hp.ang2pix(nside=2048,theta=(np.pi/2) - b,phi=l)
    EBV = EBVmap[index]

    #Calculate Av and A_lambda:
    Av = Rv * EBV
    A_lambda = Av * CCM(lamb)

    return A_lambda, EBV


def extinction_corr(spectra,lambdas,ra,dec):
    '''
    
    Corrects spectra for the effects of galactic extinction.
    
    Input: Wavelenghts, Fluxes, RA, Dec
    Returns: Fluxes corrected for the effects of Milky Way dust.

    '''
    
    #Turn RA, Dec into galactic coordinates:
    l, b = wcs.get_galactic_coordinates(ra,dec)
    
    #Get the extinction in each wavelenght:
    A_lambdas, EBV = calc_extinction(l,b,lambdas)
    
    #Calculate corrected spectra:
    corr_spectra = spectra * np.exp(A_lambdas/np.log(10))
    
    return corr_spectra