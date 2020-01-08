from __future__ import print_function

from os import path
import glob

import sys
import numpy as np
from scipy import ndimage
from astropy.io import fits

import util as util
import ssp
import scipy.integrate as integrate

def exp_SFH(tage,tau=1.,t0=13.7, norm=1.):
    tt=t0-tage
    if tt<0:
        SFR=0
    else:
    #pdb.set_trace()
        SFR=norm*np.exp(-1.*tt/tau)
    return SFR

def simple_SFH(tage, tau=1.,t0=13.7, norm=1.):
    dt=np.abs(tage-t0)
    #pdb.set_trace()
    if dt<tau:
        SFR=1
    else:
        SFR=0
    return SFR

def sfh_to_ssp(SFH,fage,**SFH_kwargs):

    Nage=len(fage)
    tage=np.zeros(Nage+1)
    tage[0]=0.
    for i in range(Nage-1):
        tage[i+1]=(fage[i]+fage[i+1])/2.
    tage[Nage]=2*fage[Nage-1]-tage[Nage-1]

    fssp=np.zeros(Nage)

    #pdb.set_trace()
    for i in range(Nage):
        fssp[i]=integrate.quad(lambda x: SFH(x,**SFH_kwargs),tage[i],tage[i+1])[0]

    return fssp

from scipy import special, fftpack

###############################################################################
# NAME:
#   EMLINE
#
# MODIFICATION HISTORY:
#   V1.0.0: Written using analytic pixel integration.
#       Michele Cappellari, Oxford, 10 August 2016
#   V2.0.0: Define lines in frequency domain for a rigorous
#       convolution within pPXF at any sigma. 
#       Introduced `pixel` keyword for optional pixel convolution.
#       MC, Oxford, 26 May 2017

def emline(logLam_temp, line_wave, FWHM_gal, pixel=True):
    """
    Instrumental Gaussian line spread function (LSF), 
    optionally integrated within the pixels. The function 
    is normalized in such a way that
    
            line.sum() = 1
    
    When the LSF is not severey undersampled, and when 
    pixel=False, the output of this function is nearly 
    indistinguishable from a normalized Gaussian:
    
      x = (logLam_temp - np.log(line_wave))/dx
      gauss = np.exp(-0.5*(x/xsig)**2)
      gauss /= np.sqrt(2*np.pi)*xsig

    However, to deal rigorously with the possibility of severe 
    undersampling, this Gaussian is defined analytically in 
    frequency domain and transformed numerically to time domain. 
    This makes the convolution exact within pPXF regardless of sigma.
    
    :param logLam_temp: np.log(wavelength) in Angstrom
    :param line_wave: Vector of lines wavelength in Angstrom
    :param FWHM_gal: FWHM in Angstrom. This can be a scalar or the
        name of a function wich returns the FWHM for given wavelength.
    :param pixel: set to True to perform integration over the pixels.
    :return: LSF computed for every logLam_temp

    """
    if callable(FWHM_gal):
        FWHM_gal = FWHM_gal(line_wave)

    n = logLam_temp.size
    npad = fftpack.next_fast_len(n)
    nl = npad//2 + 1  # Expected length of rfft

    dx = (logLam_temp[-1] - logLam_temp[0])/(n - 1)
    x0 = (np.log(line_wave) - logLam_temp[0])/dx
    xsig = FWHM_gal/2.355/line_wave/dx    # sigma in pixels units
    w = np.linspace(0, np.pi, nl)[:, None]

    # Gaussian with sigma=xsig and center=x0,
    # optionally convolved with an unitary pixel UnitBox[]
    # analytically defined in frequency domain
    # and numerically transformed to time domain
    rfft = np.exp(-0.5*(w*xsig)**2 - 1j*w*x0)
    if pixel:
        rfft *= np.sinc(w/(2*np.pi))
    line = np.fft.irfft(rfft, n=npad, axis=0)

    return line[:n, :]

###############################################################################
# NAME:
#   EMISSION_LINES
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Oxford, 7 January 2014
#   V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
#   V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the PPXF fit
#       which can cause numerical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014
#   V1.2.0: Perform integration over the pixels of the Gaussian line spread function
#       using the new function emline(). Thanks to Eric Emsellem for the suggestion.
#       MC, Oxford, 10 August 2016
#   V1.2.1: Allow FWHM_gal to be a function of wavelength. MC, Oxford, 16 August 2016

def emission_lines(logLam_temp, lamRange_gal, FWHM_gal):
    """
    Generates an array of Gaussian emission lines to be used as gas templates in PPXF.
    These templates represent the instrumental line spread function (LSF) at the
    set of wavelengths of each emission line.

    Additional lines can be easily added by editing the code of this procedure,
    which is meant as a template to be modified by the users where needed.

    For accuracy the Gaussians are integrated over the pixels boundaries.
    This integration is only useful for quite unresolved Gaussians but one should
    keep in mind that, if the LSF is not well resolved, the input spectrum is not
    properly sampled and one is wasting useful information from the spectrograph!

    The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    :param logLam_temp: is the natural log of the wavelength of the templates in
        Angstrom. logLam_temp should be the same as that of the stellar templates.
    :param lamRange_gal: is the estimated rest-frame fitted wavelength range
        Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
        where wave is the observed wavelength of the fitted galaxy pixels and
        z is an initial rough estimate of the galaxy redshift.
    :param FWHM_gal: is the instrumantal FWHM of the galaxy spectrum under study
        in Angstrom. One can pass either a scalar or the name "func" of a function
        func(wave) which returns the FWHM for a given vector of input wavelengths.
    :return: emission_lines, line_names, line_wave

    """

    # Balmer Series:      Hdelta   Hgamma    Hbeta   Halpha
    line_wave = np.array([4101.76, 4340.47, 4861.33, 6562.80])  # air wavelengths
    line_names = np.array(['Hdelta', 'Hgamma', 'Hbeta', 'Halpha'])
    emission_lines = emline(logLam_temp, line_wave, FWHM_gal)

    #                 -----[OII]-----    -----[SII]-----
    lines = np.array([3726.03, 3728.82, 6716.47, 6730.85])  # air wavelengths
    names = np.array(['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731'])
    gauss = emline(logLam_temp, lines, FWHM_gal)
    emission_lines = np.append(emission_lines, gauss, 1)
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #                 -----[OIII]-----
    lines = np.array([4958.92, 5006.84])    # air wavelengths
    doublet = 0.33*emline(logLam_temp, lines[0], FWHM_gal) + emline(logLam_temp, lines[1], FWHM_gal)
    emission_lines = np.append(emission_lines, doublet, 1)
    line_names = np.append(line_names, '[OIII]5007d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #                  -----[OI]-----
    lines = np.array([6300.30, 6363.67])    # air wavelengths
    doublet = emline(logLam_temp, lines[0], FWHM_gal) + 0.33*emline(logLam_temp, lines[1], FWHM_gal)
    emission_lines = np.append(emission_lines, doublet, 1)
    line_names = np.append(line_names, '[OI]6300d') # single template for this doublet
    line_wave = np.append(line_wave, lines[0])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #                 -----[NII]-----
    lines = np.array([6548.03, 6583.41])    # air wavelengths
    doublet = 0.33*emline(logLam_temp, lines[0], FWHM_gal) + emline(logLam_temp, lines[1], FWHM_gal)
    emission_lines = np.append(emission_lines, doublet, 1)
    line_names = np.append(line_names, '[NII]6583d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the pPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

    #print('Emission lines included in gas templates:')
    #print(line_names)

    return emission_lines, line_names, line_wave

def gas_spec(logLam_temp,velscale,FWHM_gal):
    lamRange_gal=np.exp([logLam_temp.min(),logLam_temp.max()])
    emlines,line_names,line_wave=emission_lines(logLam_temp,lamRange_gal,FWHM_gal)
    
    # Flux Ratio, normalized at Ha, namely F(Ha)=1
    #ratio=np.array([209,428,1166,8096,866,674,1192,1008,1304,268,5516])/8096. # NGC2500
    ratio=np.array([254,480,1000,2713,1667,2450,378,272,3207,87,315])/1000. # IIZw072
    emlines=emlines*ratio
    gas_spectras=np.copy(emlines)
    ha_flux=np.sum(emlines[:,3])
    return gas_spectras,ha_flux

class ModelSpectra():
    
    def __init__(self,SFH,ssp,velscale=70.,FWHM_int=3.,vdisp=200.,FeH=0.,EW_Ha=-99., \
            lam_range=[3500,10000],vel=0.,**SFH_kwargs):

        """
        Produces an model galaxy spectrum with any assumed FWHM,vdisp and EW_Ha
        Thie script relies on the bc03 template file I generated in fits format 

        :param SFH: star formation history, function, *args,**kargs
        :param velscale: desired velocity scale for the output templates library in km/s
            (e.g. 60). This is generally the same or an integer fraction of the velscale
            of the galaxy spectrum.
        :param FWHM_gal: vector or scalar of the FWHM of the instrumental resolution of
            the galaxy spectrum in Angstrom. (default 3A)
        :param vdisp: velocity dispersion of stellar components (km/s)
        :param FeH: stellar metalicity [Fe/H]
        :param line_width: emission line width (sigma of Gaussion, km/s)
        :param EW_Ha: EW of Ha, if not assigned, EW_Ha is estimated from SSFR, which is estimated from SFH
        :param lam_range: wavelength range,default [3600,9200]A
        :param **SFH_args, parameters of SFH function
        """
        
        
        z=vel/300000.
        lam_range=np.array(lam_range)
        FWHM_gal=vdisp*2.355/velscale
        FWHM_obs=np.sqrt(FWHM_int**2+FWHM_gal**2)
        
        AM_tpl=ssp.templates

        # select metal bins
        metals=ssp.metal_grid[0,:]
        minloc=np.argmin(abs(FeH-metals))
        tpls=AM_tpl[:,:,minloc]
        fmass=ssp.fmass_ssp()[:,minloc]

        #age_bins
        ages=ssp.age_grid[:,0]
        fssp=sfh_to_ssp(SFH,ages,**SFH_kwargs)
        mass=np.dot(fmass,fssp)
        Stellar0=np.dot(tpls,fssp)*mass
        
        # Broadening
        FWHM_dif = np.sqrt(FWHM_obs**2 - ssp.FWHM**2) + 0.0001
        sigma_dif = FWHM_dif/2.355/ssp.CDELT1         # Sigma difference in pixels
        
        Stellar = ndimage.gaussian_filter1d(Stellar0, sigma_dif)

        #bc03 is in air_wave
        wave=np.exp(ssp.log_lam_temp)*(1+z)

        # Ionized Gas Spectra
        Gases, ha_flux = gas_spec(ssp.log_lam_temp, velscale, FWHM_obs)
        
        # Normalized by EW(Ha)
        
        sel=np.where(Gases[:,3]>0.01)
        # average continum 
        fcont=np.median(Stellar[sel])
        
        if EW_Ha>0:
            fem=EW_Ha*fcont/ha_flux
            Gases=Gases*fem
            Gas=np.sum(Gases,axis=1)
        else:
            Gas=np.sum(Gases,axis=1)*0
        
        Spectra=Stellar+Gas
        
        # Trim Spetral
        itrim=np.logical_and(wave>=lam_range[0],wave<=lam_range[1])
    
        self.wave=wave[itrim]
        self.stellar=Stellar[itrim]
        self.gas=Gas[itrim]
        self.emlines=Gases[itrim,:]
        self.flux=Spectra[itrim]
        self.z=z
        self.stellarmass=mass
        
    def add_AGN(self,Atype,fBH=0.003,Edd_ratio=1.):
        
        #BH mass: fBH*1Msolar
        #AGN bol luminosity: in unit of Lsun
        Lbol=fBH*Edd_ratio*1.3e5/3.826*self.stellarmass
        #from Heckman 2004
        L5000=Lbol/10.9

        # AGN templates
        if Atype == 1:
            # 3C273
            filename='./data/AGN_template/3C273.dat'
            wave,flux=util.readcol(filename,comments='#',usecols=(0,2))
            wave=wave*1e4
        elif Atype==2:
            # PKS 1345+12
            filename='./data/AGN_template/PKS-1345+12.dat'
            wave,flux=util.readcol(filename,comments='#',usecols=(0,2))
            wave=wave*1e4
        else:
            raise Exception('wrong AGN type')
            
        sel=(wave/(1+self.z)>4987)&(wave/(1+self.z)<5010)
        f5000=np.mean(flux[sel])
        fnorm=L5000/f5000
        
        AGN_emission=fnorm*flux
        wave0=self.wave/(1+self.z)
        AGN_flux=np.interp(wave0,wave,AGN_emission)
        
        self.AGN=AGN_flux
        self.flux=self.gas+self.stellar+self.AGN
        
    def remove_AGN(self):
        self.AGN=np.zeros(len(self.gas))
        self.flux=self.gas+self.stellar
        
    def output_spectra(self,specname):
        from astropy.table import Table
        from astropy.table import Column
        wave=self.wave
        spectra=self.flux
        stellar=self.stellar
        gas=self.gas
        
        c1 = fits.Column(name='WAVELENGTH', array=wave, format='E', unit='angstrom')
        c2 = fits.Column(name='FLUX', array=flux, format='E', unit='flam')
        c3 = fits.Column(name='STELLAR', array=stellar, format='E', unit='flam')
        c4 = fits.Column(name='GAS', array=gas, format='E', unit='flam')
        if hasattr(self,'AGN'):
            c5 = fits.Column(name='AGN', array=self.AGN, format='E', unit='flam')
        
        cols = fits.ColDefs([c1, c2, c3, c4])
        hdu = fits.BinTableHDU.from_columns(cols)
            
        hdr=fits.Header()
        
        hdu.writeto(specname)
        
class GalTemp(object):
    def __init__(self, Gtype, velscale=50, vdisp=-99, FWHM_int=3., lam_range=[3500,10000], vel=0.):
        
        if Gtype.capitalize() in ['S0', 'Sa', 'Sb', 'Sc', 'Im']:
            filename='GalTemp-'+Gtype.capitalize()+'.fits'
            hdulist=fits.open(filename)
            data=hdulist[1].data
            wave=data['WAVELENGTH']
            flux=data['Flux']
            
            # Broadening
            
            FWHM_temp=2.5
            if vdisp>0:
                FWHM_gal=vdisp*2.36/velscale
                FWHM_obs=np.sqrt(FWHM_int**2+FWHM_gal**2)
            else:
                FWHM_obs=FWHM_int
            if FWHM_obs>FWHM_temp:
                sigma=np.sqrt(FWHM_obs**2 - FWHM_temp**2)
                flux=ndimage.gaussian_filter1d(flux,sigma)
                
            # Redshift
            
            z=vel/3e5
            lam_range_dered=np.array(lam_range)/(1+z)
            ind_dered=np.logical_and(wave>lam_range_dered[0],wave<lam_range_dered[1])
            wave_new=wave[ind_dered]
            flux_new=flux[ind_dered]
            specNew, logLam, _=util.log_rebin(lam_range, flux_new, velscale=velscale)
            
            self.flux=specNew
            self.wave=np.exp(logLam)
        else:
            raise ValueError("Gtype must be included in 'S0', 'Sa', 'Sb', 'Sc', 'Im'")