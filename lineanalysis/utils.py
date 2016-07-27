
#import spectral_cube as sp
#import radio_beam
from radio_beam import Beam
#import astropy as ap
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.table import Table
#from astroquery import splatalogue
#from viridis import viridis
#import matplotlib.gridspec as gridspec
#from astropy.table import Table
#from astropy.coordinates import SkyCoord
#from wcsaxes import datasets, WCS
from scipy.optimize import curve_fit
import os

"""
Yeah this is wrong, beam dilution and beam filling is the same.
Or at least ppl use it interchangable
Beam Filling : apply when calculating Flux from Column density

Beam Dilution: apply when calculating Column density from Flux

"""
#TODO: or are they the same?

def get_partition_file(part_file_directory = '/home/magnusp/work/data/partition/new/', 
        part_file = 'hdco.dat', get_raw=False):
    #
    from scipy.optimize import leastsq as _ls
    #from scipy import loadtxt as _loadtxt

    # now load the file
    t,q = np.loadtxt(os.path.join(part_file_directory, part_file),
                 dtype='str').T

    import sys
    # different handling of strings in Python 2 vs. 3
    if sys.version_info >= (3, 0):
        t = np.flipud(np.array([float((i[4:-1]).replace("_", ".")) for i in t]))
        q = np.flipud(np.array([float(i[2:-1].strip()) for i in q]))
    else:
        t = np.flipud(np.array([float((i[2:]).replace("_", ".")) for i in t]))
        q = np.flipud(np.array([float(i.strip()) for i in q]))
    


    fitfunc = lambda p,t: q - (p[0]*t**p[1])
    # initial guess...
    p0 = [0.05,1.5] 

    # now least square fitting
    p_fit = _ls(fitfunc, p0, args=(t))[0]

    # create the function to get the partition function value for 
    # given temperature
    qrot = lambda x: p_fit[0] * x**p_fit[1]
    if get_raw:
        return qrot, t, q
    return qrot

def get_partition_cdms(
                    partfunc_url="http://www.astro.uni-koeln.de/site/vorhersagen/catalog/partition_function.html",
                    ):
    from urllib.request import urlopen
    import pandas as pd
    r = [i.decode("utf-8") for i in urlopen(partfunc_url).readlines()]
    start = r.index("<small><pre>\n")
    stop = r.index("</pre></small>\n")
    # header names
    part_header = r[start+1][:-1]
    part_header = part_header.split()
    # table
    part_table = r[start+3:stop]
    part_table = "".join(part_table).split('\n')[:-1]

    # need to clean up table
    filt1 = lambda x : [s.strip() for s in x.split('  ') if s]
    part_table = [filt1(i) for i in part_table]
    #print(part_header)
    filt2 = lambda x: x[0].split()
    names = [filt2(i)[1]  for i in part_table]
    ids = [filt2(i)[0]  for i in part_table]
    part_table = pd.DataFrame(part_table, 
                              index=ids,
                             columns=part_header[1:],
                             )
    part_table.replace('---', 'nan', inplace=True)
    part_table.insert(0,'species', names)
    #print(part_table.head())

    qs = part_table[part_header[3:]].values.astype('float')
    ts = [float(i.strip('lg(Q(' ).strip('))')) for i in part_header[3:]]
    qnt = dict( zip(names,qs) )
    return ts, qnt


#@u.quantity_input(bmin=u.arcsec,bmaj=u.arcsec,smin=u.arcsec,smaj=u.arcsec)
def beam_filling(bmin,bmaj,smin,smaj):
    bf = bmin*bmaj / (bmin*bmaj + smin*smaj)
    return bf

def beam_dilution(bmin,bmaj,smin,smaj):
    """
    Source / Beam
    """
    #bd = bmin*bmaj / (smin*smaj)
    bd = smin*smaj / (bmin*bmaj)
    return bd

@u.quantity_input(ntot=u.cm**-2, eu=u.Kelvin, tex=u.Kelvin)
def calc_nup_tex(ntot=None, gup=None, qrot=None, eup=None, tex=None):
    nup = ntot /  qrot * gup *  np.exp(-1*eup/tex)
    return nup

@u.quantity_input(fwhm=u.km/u.s, nu=u.Hz , nup=u.cm**-2, aul=u.s**-1, eup=u.Kelvin, tex=u.Kelvin)
def calc_tau(fwhm=None, nu=None , nup=None, aul=None, eup=None, tex=None):
    tau = nup * aul * c.c**3 / (8 * np.pi * c.h * nu**3) * c.h/(fwhm)  * ( np.exp(c.h*nu/(c.k_B*tex))-1 )
    return tau

@u.quantity_input(aul=u.s**-1, 
                  nu=u.Hz, 
                  ntot=u.cm**-2, 
                  eup=u.Kelvin, 
                  tex=u.Kelvin, 
                  fwhm=u.km/u.s)
def calc_intensity(aul=None, 
                   nu=None,
                   ntot=None, 
                   qrot=None,
                   gup=None, 
                   eup=None, 
                   tex=None, 
                   fwhm=None,
                   beam=None, 
                   source=None,
                   usetau=False):
    # Nup, column densit of upper levelr
    nup = calc_nup_tex(ntot=ntot, gup=gup, qrot=qrot(tex.value), eup=eup, tex=tex)
    # optical depth at line peak
    tau = calc_tau(fwhm=fwhm, nu=nu, nup=nup, aul=aul, eup=eup, tex=tex)
    tau = tau.decompose().value
    # calculate the beam filling factor to be used
    #bf = beam_filling(bmin=beam[0], bmaj=beam[1], smin=source[0], smaj=source[1])
    bd = beam_dilution(bmin=beam[0], bmaj=beam[1], smin=source[0], smaj=source[1])
    # the optical depth effect on the intensity
    c_tau = tau / (1 - np.exp(-tau))
    w = nup * c.h * c.c**3 * aul / (8. * np.pi * c.k_B * nu**2 * bd )#* c_tau)
    # old with beam filling, I think it is wrong.
    #w = nup * c.h * c.c**3 * aul / (8. * np.pi * c.k_B * nu**2 * bf )#* c_tau)
    # alternative, gives the same
    #w_alt = ntot * c.h * c.c**3 * aul * gup / (8. * np.pi * c.k_B * nu**2 * qrot(tex.value) * np.exp(eup/tex) * bf)
    if usetau:
        w /= c_tau
    return w.to(u.K * u.km/u.s), tau

@u.quantity_input(aul=u.s**-1, 
                  nu=u.Hz, 
                  w=u.K * u.km / u.s, 
                  eup=u.Kelvin, 
                  tex=u.Kelvin,
                  fwhm=u.km/u.s,
                  nup=u.cm**-2,
                  )
def calc_ntot_obs(nu=None, qrot=None, aul=None, gup=None, eup=None, 
            tex=None, w=None, beam=None, source=None, fwhm=None, 
            usetau=False):
    # beam dilution factor
    bd = beam_dilution(bmin=beam[0], bmaj=beam[1], smin=source[0], smaj=source[1])
    # upper energy level column density, for optical depth estimate
    nup = calc_nup_obs(nu=nu, 
                aij=aul, 
                w=w/bd,
                )
    # apply beam dilution factor
    # nu \propto w (source/beam)**-1
    # calculate optical depth, tau
    tau = calc_tau(fwhm=fwhm, nu=nu , nup=nup, aul=aul, eup=eup, tex=tex)
    tau = tau.decompose()
    c_tau = tau / (1 - np.exp(-tau))
    # now we can estimate the column densities
    # divide by the beam dilution factor
    up = (8 * np.pi * c.k_B * nu**2 * qrot * np.exp(eup/tex) * w / bd)
    down = (c.h * c.c**3 * aul * gup)
    ntot = up/down
    if usetau:
        ntot *= c_tau
    return ntot.to(u.cm**-2)

@u.quantity_input(intensity=u.Jy*u.km/u.s, fwhm=u.km/u.s, pos=u.km/u.s)
def get_gaussian(intensity, fwhm, pos, tau=0):
    sigma = fwhm / (2 * (2*np.log(2))**.5 )
    amplitude = intensity / (sigma * (2 * np.pi)**.5 )
    gauss = lambda x: amplitude * np.exp(-(x-pos)**2/(2 * sigma**2))
    return gauss

@u.quantity_input(nu=u.GHz, aij=1/u.s, w=u.K*u.km/u.s)
def calc_nup_obs(nu=None, 
                aij=None, 
                w=None):
    # beam filling factor
    #bf = beam_filling(bmin=beam[0], bmaj=beam[1], smin=source[0], smaj=source[1])
    # calculate nup    
    nup_obs = 8 * np.pi * c.k_B * nu**2 / (c.h * c.c**3 * aij)
    nup_obs *= w
    return nup_obs.decompose().to(1/u.cm**2)

@u.quantity_input(ilims=u.km/u.s)
def integrate_line(linedata=None,
                spectra=None,
                lnum=None,
                ilims=None,
                spos=None,
                velrange=5,
                vsys=None,
                ):
    vel_lims = linedata['vel'].quantity[lnum] + np.array([-velrange, velrange])*u.km/u.s + vsys
    vel_mask = (spectra['vel']>vel_lims[0]) * (spectra['vel']<vel_lims[1])
    xvalues = spectra['vel'][vel_mask] - linedata['vel'][lnum]
    xdiff = abs( np.diff(xvalues)[0] )
    xwidth = xdiff * u.km/u.s
    yvalues = spectra[spos][vel_mask]
    # define selection over which to integrate
    if np.diff(xvalues)[0] <0:
        ilims_bool = ((xvalues.quantity + xwidth/2.)>=ilims[0]) * ((xvalues.quantity - xwidth/2.)<=ilims[1])
    else:
        ilims_bool = ((xvalues.quantity - xwidth/2.)>=ilims[0]) * ((xvalues.quantity + xwidth/2.)<=ilims[1])
    obs_intensity = yvalues[ilims_bool].quantity.sum() * abs(np.diff(xvalues.quantity)[0])
    return obs_intensity.to(u.Jy * u.km/(u.beam * u.s))

def fit_rotd(linedata=None,
            qrot=None,
            lnums=None,
            ):
    # y data is nup/gup
    y = linedata['nup'].quantity[lnums] / linedata['gup'][lnums]
    y = np.log(y.value)
    x = linedata['eup'].quantity[lnums]

    from scipy.optimize import curve_fit
    f_line = lambda x,a,b: a+b*x
    par, cov = curve_fit(f_line, x,y, sigma=y*0.1)
    err = np.diag(cov)**0.5

    #plt.scatter(x,y)
    #plt.plot([30,300], f_line(np.array([30,300]), par[0],par[1]))

    tex = -1/par[1]
    #dtex = err[1]

    ntot = np.exp(par[0]) * qrot(tex)
    #dntot = np.exp(err[0])*qrot(tex)
    #print(tex,dtex,ntot,dntot)
    
    print('Tex ', tex, 'Ntot ', ntot)
    
    return x,y,par,f_line, tex, ntot

@u.quantity_input(vsys=u.km/u.s)
def get_line_data(linedata=None,
            spectra=None,
            name='no-name-given',
            qrot=None,  # this is now the function!
            ilims=None,
            lnums=None,
            beam=None, 
            source=None,
            spos=None,
            vsys=None,
            velrange=None,
            ):
    beamobj = Beam(beam[0]*u.arcsec ,beam[1]*u.arcsec, beam[2]*u.deg)
    bd = beam_dilution(bmin=beam[0], bmaj=beam[1], smin=source[0], smaj=source[1])
    nup = Table.Column(length=len(linedata['freq']), unit=1/u.cm**2) * np.nan
    w_jy = Table.Column(length=len(linedata['freq']), unit=u.Jy * u.km/u.s) * np.nan
    w_k = Table.Column(length=len(linedata['freq']), unit=u.K * u.km/u.s) * np.nan
    for i in range(len(lnums)):
        # integrate to get the observed intensity
        wi_jy = integrate_line(linedata=linedata,
                spectra=spectra,
                lnum=lnums[i],
                ilims=ilims[i],
                spos=spos,
                velrange=5.,
                vsys=vsys)
        # remove the /beam in the integrated intensity
        # works for this data, but necessarly for all
        wi_jy *= u.beam
        # convert to K
        wi_k = wi_jy * beamobj.jtok(linedata['freq'].quantity[i])/u.Jy
        # correct for beam dilution
        wi_k /= bd 
        nupi = calc_nup_obs(nu=linedata['freq'].quantity[i], 
                aij=10**linedata['aij'][i]/u.s, 
                w=wi_k,
                #beam=beam, 
                #source=source,
                )
        print(nupi,wi_jy,wi_k)
        nup[lnums[i]] = nupi.to(1/u.cm**2).value
        w_jy[lnums[i]] = wi_jy.to(u.Jy * u.km/u.s).value
        w_k[lnums[i]] = wi_k.to(u.K * u.km/u.s).value
        
    linedata['nup'] = nup
    linedata['w_jy'] = w_jy
    linedata['w_k'] = w_k
    #
    
    #fit_rotd(linedata=linedata,
    #        qrot=qrot,
    #        lnums=lnums)
    #        lnums=lnums[i],
    #        qrot=qrot
    #        )
    # first integrate to get the line intensities
    #ws = [integrate_line(linedata[x], ilims) for x in lnums]
    # from the line intensities, calculate Nu 
    #nus = calc_nup_obs()
    # fit the rotational diagram
    #fit_rotd()
    
    return linedata

def calc_ntot_crit(taulim=1.):
    return None
