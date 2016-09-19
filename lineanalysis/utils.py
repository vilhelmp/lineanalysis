
import radio_beam
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.table import Table, Column
from scipy.optimize import curve_fit
import os
import sys

"""
Beam Dilution and Beam Filling are the same thing.
O_s=source solid angle
O_b = beam solid angle
When O_s >=~ O_b:
    bd = O_s/ (O_b + O_s)
When O_s << O_b: (approximation)
    bd = O_s/O_b
Ntot \propto W/bd
Nup = \propto W/bd
i.e.
W \propto Ntot * bd
W \propto Nup * bd

"""


#################################################
#           PARTITION FUNCTION PARSING          #
#################################################

DEFAULT_PART_FILE_DIRECTORY = '/home/magnusp/work/data/partition/new/'

# Read and parse partition function from file
def get_partition_file(part_file_directory = DEFAULT_PART_FILE_DIRECTORY,
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

# Get the partition function from the CDMS online
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

#################################################
#                   OTHER                       #
#################################################


def diffarr(arr, m=2):
    """
    :param arr: Input array with numbers
    :param m: how many steps ahead to difference
    Returns arr[i+m] - arr[i] for each element i
    Return array of length len(arr)-m
    """
    return arr[m:] - arr[:-m]


# Calculate the beam dilution factor
def beam_dilution(bmin,bmaj,smin,smaj):
    """
    Source / Beam
    """
    #bd = bmin*bmaj / (smin*smaj)
    bd = smin*smaj / (smin*smaj + bmin*bmaj)
    return bd

# Calculate Nu from Ntot and Tex
@u.quantity_input(ntot=u.cm**-2, eu=u.Kelvin, tex=u.Kelvin)
def calc_nup_tex(ntot=None, gup=None, qrot=None, eup=None, tex=None):
    nup = ntot /  qrot * gup *  np.exp(-1*eup/tex)
    return nup

# Calculate tau from Nu and Tex
@u.quantity_input(fwhm=u.km/u.s, nu=u.Hz , nup=u.cm**-2, aul=u.s**-1, eup=u.Kelvin, tex=u.Kelvin)
def calc_tau(fwhm=None, nu=None , nup=None, aul=None, eup=None, tex=None):
    tau = nup * aul * c.c**3 / (8 * np.pi * c.h * nu**3) * c.h/(fwhm)  * ( np.exp(c.h*nu/(c.k_B*tex))-1 )
    return tau

# Calculate Ntot from integrated flux and Tex
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
    """
    Calculates the column density of the species, using the input flux.
    If 'usetau' is set equal True, the optical depth correction factor is included like
    c_t = tau / (1 - exp(-tau))
    where tau is calculated from the column density of the upper energy level (using the calc_nup_obs
    taking as input [beam dilution corrected] flux, line frequency and Einstein A coefficient)
    The calculations corrects for beam dilution, i.e. the inpu flux, 'w' is divided by the
    beam dilution w/bd where bd = source^2 / (beam^2 + source^2).

    :param nu:
    :param qrot:
    :param aul:
    :param gup:
    :param eup:
    :param tex:
    :param w:
    :param beam:
    :param source:
    :param fwhm:
    :param usetau:
    :return:
    """
    # beam dilution factor
    bd = beam_dilution(bmin=beam[0], bmaj=beam[1], smin=source[0], smaj=source[1])
    # upper energy level column density, for optical depth estimate
    w_bdcorr = w/bd
    #print('Input flux corrected for beam dilution.')
    nup = calc_nup_obs(nu=nu, 
                aij=aul, 
                w=w_bdcorr,
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

# Calculate Nup from integrated flux and Tex
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

# Calculate Nlo from integrated flux and Tex
@u.quantity_input(nu=u.GHz, aij=1/u.s, w=u.K*u.km/u.s)
def calc_nlo_obs(nu=None,
                aij=None,
                w=None):
    #TODO this has not been fixed!
    nlo_obs = 8 * np.pi * c.k_B * nu**2 / (c.h * c.c**3 * aij)
    nlo_obs *= w
    return nlo_ki.decompose().to(1/u.cm**2)



# Make a Gaussian function from integrated total intensity, fwhm
# and position (x-coord center)
@u.quantity_input(intensity=u.Jy*u.km/u.s, fwhm=u.km/u.s, pos=u.km/u.s)
def get_gaussian(intensity, fwhm, pos, tau=0):
    sigma = fwhm / (2 * (2*np.log(2))**.5 )
    amplitude = intensity / (sigma * (2 * np.pi)**.5 )
    gauss = lambda x: amplitude * np.exp(-(x-pos)**2/(2 * sigma**2))
    return gauss

# OLD function to integrate the line flux
@u.quantity_input(ilims=u.km/u.s, velrange=u.km/u.s)
def integrate_line_old(linedata=None,
                spectra=None,
                lnum=None,
                ilims=None,
                spos=None,
                velrange=5.*u.km/u.s,
                vsys=None,
                ):
    """
    :param linedata: Molecular line data (from CDMS), Astropy Table
    :param spectra: Observed spectra, Astropy Table :
    :param lnum: Line number (in Molecular line data) to integrate : [integer] or None : No unit
    :param ilims: Integration limits : [start, stop] : km/s or equivalent units
    :param spos: Name of the flux column of the spectra table : str : No unit
    :param velrange: Limit for the slice around the line, should be larger than ilims (smaller memory footprint?) : float : km/s
    :param vsys: Systemic velocity to correct so that the line is at 0 km/s
    """
    vel_lims = linedata['vel'].quantity[lnum] + np.array([-velrange, velrange]) + vsys
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

# Function to fit line in rotational diagram
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

# OLD Function to calculate line intensities from observations
# replaced by "integrate_all_lines"
@u.quantity_input(vsys=u.km/u.s)
def get_line_data_old(linedata=None,
            spectra=None,
            name='no-name-given',
            ilims=None,
            lnums=None,
            beam=None, 
            source=None,
            spos=None,
            vsys=None,
            velrange=None,
            ):
    """Function to calculate line intensities from observations
    lnums can be several lines, i.e [1,10].
    If lnums=None then all lines are used"""
    beamobj = radio_beam.Beam(beam[0]*u.arcsec ,beam[1]*u.arcsec, beam[2]*u.deg)
    bd = beam_dilution(bmin=beam[0], bmaj=beam[1], smin=source[0], smaj=source[1])
    nup = Table.Column(length=len(linedata['freq']), unit=1/u.cm**2) * np.nan
    w_jy = Table.Column(length=len(linedata['freq']), unit=u.Jy * u.km/u.s) * np.nan
    w_k = Table.Column(length=len(linedata['freq']), unit=u.K * u.km/u.s) * np.nan
    if lnums==None:
        # if lnums is not given,
        # take all lines
        print('Integrating flux for all lines')
        lnums=np.arange(len(linedata))
    for i in range(len(lnums)):
        # integrate to get the observed intensity
        wi_jy = integrate_line(linedata=linedata,
                spectra=spectra,
                lnum=lnums[i],
                ilims=ilims[i],
                spos=spos,
                velrange=velrange,
                vsys=vsys)
        # remove the /beam in the integrated intensity
        # works for this data, but necessarly for all
        wi_jy *= u.beam
        # convert to K
        wi_k = wi_jy * beamobj.jtok(linedata['freq'].quantity[i])/u.Jy
        # correct for beam dilution
        #wi_k /= bd
        print('Not corrected for beam dilution ')
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

    return linedata

def calc_ntot_crit(taulim=1.):
    return None


#################################################
#                OBSERVATIONS                   #
# FUNCTIONS TO INTEGRATE THE OBSERVED SPECTRUM  #
#################################################

@u.quantity_input(ilims=u.km/u.s, vsys=u.km/u.s)
def integrate_line(linedata=None,
                   spectra=None,
                   ilims=None,
                   vsys=None,
                   fluxcol=None,
                   dbg=False):
    '''
    Function to calculate one integrated intensity from observations.
    :param linedata:
    :param spectra:
    :param ilims:
    :param vsys:
    :param fluxcol:
    :param dbg: debug mode, modifies what is returned
    :return: returns the integrated observed intensity
             if dbg=True it will return xvalues, yvalues, xwidth, ilims_bool, obs_intensity
    '''
    xvalues = spectra['vel_obs'] - linedata['vel_vsys']
    xdiff = np.diff(xvalues)[0]
    xwidth = abs(xdiff) * u.km/u.s
    if xdiff<0:
        ilims_bool = ((xvalues.quantity + xwidth/2.)>=ilims[0]) * ((xvalues.quantity - xwidth/2.)<=ilims[1])
    else:
        ilims_bool = ((xvalues.quantity - xwidth/2.)>=ilims[0]) * ((xvalues.quantity + xwidth/2.)<=ilims[1])
    yvalues = spectra[fluxcol]
    obs_flux = yvalues[ilims_bool].quantity.sum() * xwidth
    if dbg:
        return xvalues, yvalues, xwidth, ilims_bool, obs_flux
    return obs_flux


@u.quantity_input(ilims=u.km/u.s, vsys=u.km/u.s)
def integrate_all_lines(linedata=None,
                        model=None,
                        spectra=None,
                        ilims=None,
                        vsys=None,
                        fluxcol=None,
                        dbg=False):
    '''
    Function to calculate integrated intensities of observations.
    :param linedata:
    :param spectra:
    :param ilims:
    :param vsys:
    :param spos:
    :param dbg:
    :return:
    '''
    # create emtpy arry to put the integrated line strengths in
    obs_fluxes = np.array([])
    # loop to integrate the line strength of each line
    for i in range(len(linedata)):
        obs_flux = integrate_line(linedata=linedata[i:i+1],
                       spectra=spectra,
                       ilims=ilims,
                       vsys=vsys,
                       fluxcol=fluxcol,
                       dbg=False)
        obs_flux *= u.beam
        obs_fluxes = np.append(obs_fluxes,obs_flux.value)
    print('Beam in unit multiplied away.')
    # preserve unit
    obs_fluxes = obs_fluxes * obs_flux.unit
    model['W_obs'] = obs_fluxes
    return model


#################################################
#                  MODEL FLUX                   #
#     FUNCTIONS TO CALCULATE LINE STRENGTHS     #
#################################################

# Calculate the intensity of one line from Ntot and Tex
@u.quantity_input(aul=u.s**-1,
                  nu=u.Hz,
                  ntot=u.cm**-2,
                  eup=u.Kelvin,
                  tex=u.Kelvin,
                  fwhm=u.km/u.s)
def calc_line_flux(aul=None,
                   nu=None,
                   ntot=None,
                   qrot=None,
                   gup=None,
                   eup=None,
                   tex=None,
                   fwhm=None,
                   beam=None,
                   source=None,
                   usetau=True,
                   returnjy=True,
                   ):
    """
    Function to calculate the synthetic flux for one line.
    :param aul:
    :param nu:
    :param ntot:
    :param qrot:
    :param gup:
    :param eup:
    :param tex:
    :param fwhm:
    :param beam:
    :param source:
    :param usetau: apply optical depth correction: bool
    :param returnjy: return unit Jansky: bool
    :return:
    """
    # Nup, column densit of upper level, to calculate tau.
    nup = calc_nup_tex(ntot=ntot, gup=gup, qrot=qrot(tex.value), eup=eup, tex=tex)
    # optical depth at line peak, tau
    tau = calc_tau(fwhm=fwhm, nu=nu, nup=nup, aul=aul, eup=eup, tex=tex)
    tau = tau.decompose().value
    # calculate the beam dilution factor to be used
    bd = beam_dilution(bmin=beam[0], bmaj=beam[1], smin=source[0], smaj=source[1])
    # the optical depth effect on the intensity
    c_tau = tau / (1 - np.exp(-tau))
    # calculate synthetic flux, based on Ntot and Tex input.
    # gives K km/s as output
    w = nup * c.h * c.c**3 * aul / (8. * np.pi * c.k_B * nu**2 )
    # beam dilution
    w *= bd
    # if we are applying the optical depth factor
    # do it below
    if usetau:
        w /= c_tau
    if returnjy:
        beamobj = radio_beam.Beam( beam[0]*u.arcsec, beam[1]*u.arcsec, beam[2]*u.deg )
        w = (w*u.s/u.km).to(u.Jy, beamobj.jtok_equiv(nu) )
        return w* u.km/u.s, tau
    # now we want to return the correct unit
    # tau is unitless, we made sure to decompose it above
    elif not returnjy:
        return w.to(u.K * u.km/u.s), tau

def calc_line_fluxes(linedata=None,
                    model=None,
                    tex=None,
                    ntot=None,
                    species=None,
                    fwhm=None,
                    beam=None,
                    source=None,
                    usetau=True,
                    qrot_method='file',
                    part_file_directory = DEFAULT_PART_FILE_DIRECTORY,
                    returnjy=True,
                    ):
    '''
    Function to calculate synthetic line fluxes from Ntot and Tex.
    :param linedata: If only one line, send like lindedata[0:1], so that it is still a Astropy Table, and not
                     only a Row
    :param tex:
    :param ntot:
    :param species:
    :param fwhm: full widht at half maximum in km/s
    :param beam: beam size like [minor, major, pa]
    :param source: source size like [minor, major]
    :param usetau: apply the optical depth correction? : bool
        tau = nup * aul * c.c**3 / (8 * np.pi * c.h * nu**3) * c.h/(fwhm)  * ( np.exp(c.h*nu/(c.k_B*tex))-1 )
        c_tau = tau / (1 - np.exp(-tau))
        then
    :param qrot_method:
    :param part_file_directory:
    :return:
    '''
    if qrot_method.lower()=='file':
        qrot = get_partition_file(part_file_directory = part_file_directory, part_file=species.lower()+'.dat')
    else:
        raise Exception('no other qrot_method works atm')

    calc_fluxes = np.array([])
    calc_taus = np.array([])
    for i in range(len(linedata)):
        calc_flux, tau_i = calc_line_flux(aul=10**linedata['aij'][i]*u.s**-1,
                       nu=linedata['freq_rest'].quantity[i],
                       ntot=ntot,
                       qrot=qrot,
                       gup=linedata['gup'][i],
                       eup=linedata['eup'].quantity[i],
                       tex=tex,
                       fwhm=fwhm,
                       beam=beam,
                       source=source,
                       usetau=usetau,
                       returnjy=returnjy)
        calc_taus = np.append(calc_taus, tau_i)
        calc_fluxes = np.append(calc_fluxes, calc_flux.value)
    # preserve unit
    calc_fluxes = calc_fluxes * calc_flux.unit
    # now add the fluxese to a column.
    model['W_calc'] = calc_fluxes
    model['tau_calc'] = calc_taus
    model.meta['W_calc_comment'] = ['Only one beam for whole data set']
    model.meta['W_calc_beamdilution_applied'] = True
    model.meta['W_calc_info'] = dict(beam=beam, source=source)
    model.meta['W_calc_tau_applied'] = usetau
    model.meta['species'] = species
    # send the astropy table back!
    return model

# function to check table for line blends
@u.quantity_input(dnu=u.Hz)
def check_for_blend(
        model=None,
        dnu=None,
        ):
    """

    :param model:
    :param dnu: max amount of GHz away for it to be a blend
    :return:
    """
    print('Assumes sorted frequency in linedata input table.')

    # empty list to store information about blend
    nlines = len(model['freq_rest'])
    emarr = [[] for i in range(nlines)]

    for i in range(nlines):
        try:
            # check one in front
            diffnu1 = model['freq_rest'].quantity[i + 1] - model['freq_rest'].quantity[i]
            if diffnu1 <= dnu:  # BLEND!
                emarr[i].append(+1)
            # check two in front
            diffnu2 = model['freq_rest'].quantity[i + 2] - model['freq_rest'].quantity[i]
            if diffnu2 <= (dnu * 1.5):  # BLEND!
                emarr[i].append(+2)
        except(IndexError):
            break
    model['blend'] = emarr
    model.meta['blend_info'] = 'Assumes table is sorted in frequency.'
    model.meta['blend_info'] = 'blend shown by list, if [1] then line is blended with the 1 infront \
if [1,2] then blended with both 1 and 2 in front.'
    return model

# function to process the blend information of the model table
def process_model_blends(model=None,
                         action='add'):
    """
    :param model:
    :param action:
    :return:
    """
    # need to figure out how to calculate the blended fluxes
    model['W_calc_blend'] = [np.nan for i in model['W_calc']]
    model['W_calc_blend'].unit = model['W_calc'].unit
    model['tau_calc_blend'] = [np.nan for i in model['W_calc']]
    for i in range(len(model)):
        if model['blend'][i]: # if its a blend!
            # +1 because slicing rules of Python
            # this assumes that if 2 away is blended
            # # 1 away is too, i.e. again, that the table is sorted after frequency
            #-----
            # if tau was applied, we need to first un-apply it to
            # be able to sum the flux and then apply the summed tau value to the blended flux.
            tau_blends = model['tau_calc'][i:i + np.max(model['blend'][i]) + 1]
            tau_blends_sum = tau_blends.sum()
            model['tau_calc_blend'][i] = tau_blends_sum
            # if Tau was applied to original fluxes
            # do it here too
            if model.meta['W_calc_tau_applied']:
                c_tau_blends_sum = tau_blends_sum/(1 - np.exp(-tau_blends_sum))
                # first un-apply tau to the individual flux calcs
                c_tau_blends = tau_blends / (1 - np.exp(-tau_blends))
                W_calcs = model['W_calc'][i:i + np.max(model['blend'][i]) + 1].quantity
                W_calcs_notau_sum = (W_calcs*c_tau_blends).sum()
                W_sum = W_calcs_notau_sum/c_tau_blends_sum
                model['W_calc_blend'][i] = W_sum.to(model['W_calc_blend'].unit).value
                model['tau_calc_blend'][i] = tau_blends_sum
                tau_applied = True
            # if Tau was NOT applied to original fluxes,
            # do not do it here either.
            elif not model.meta['W_calc_tau_applied']:
                tau_applied = False
                W_sum = model['W_calc'][i:i + np.max(model['blend'][i]) + 1].quantity.sum()
                model['W_calc_blend'][i] = W_sum.to(model['W_calc_blend'].unit).value
                tau_applied = False
    if tau_applied:
        model.meta['W_calc_blend_tau_applied_info'] = 'Tau re-applied to model blend W sum.'
        model.meta['W_calc_blend_tau_applied'] = tau_applied
    elif not tau_applied:
        model.meta['W_calc_blend_tau_applied_info'] = 'Tau NOT re-applied to model blend W sum.'
        model.meta['W_calc_blend_tau_applied'] = tau_applied
    #
    model.meta['process_blend_info'] = 'Assumes table is sorted in frequency.'
    return model

def calc_line_fluxes_grid(linedata=None,
                              texs=None,
                              ntots=None,
                              species=None,
                              ):
    """
    Function to calculate several synthetic line fluxes from Ntot and Tex.
    :param linedata:
    :param texs:
    :param ntots:
    :param species:
    :return:
    """
    #TODO check if input has certain unit
    #TODO returns astropy Table (Tex, Ntot - row and col id)
    if qrot_method.lower()=='file':
        if part_file_directory == None:
            part_file_directory = DEFAULT_PART_FILE_DIRECTORY
        qrot = get_partition_file(part_file_directory = part_file_directory, part_file=species.lower()+'.dat')
    else:
        raise Exception('no other qrot_method works atm')

    calc_fluxes = np.array([])

    for i in range(len(linedata)):
        for t in texs:
            for n in ntots:
                calc_flux,tau_i = calc_line_flux(aul=10**linedata[i]['aij'],
                       nu=linedata[i].quantity['freq_rest'],
                       ntot=ntot,
                       qrot=qrot,
                       gup=linedata[i]['gup'],
                       eup=linedata[i].quantity['eup'],
                       tex=tex,
                       fwhm=fwhm,
                       beam=beam,
                       source=source,
                       usetau=usetau)

                calc_fluxes = np.append(calc_fluxes, calc_flux.value)
    # preserve unit
    calc_fluxes = calc_fluxes * calc_flux.unit
    return None

# function to calculate synthetic spectrum
@u.quantity_input(fwhm=u.km/u.s,
                  )
def calc_synthetic_spectrum(spectra=None,
                            model=None,
                            modelname='model_flux_1',
                            fwhm=None,
                            fluxcol=None,
                            ):
    """

    :param spectra:
    :param model:
    :param modelname:
    :param fwhm:
    :param fluxcol:

    :return:
     Returns a modified (from input) spectra table, with synthetic spectrum column
     and a separate table with the synthetic spectra snippets around each line, with
     and without blends summed.

    """
    #TODO account for blends and calculate blended spectrum
    #TODO calculate spectrum for each line
    # freq shoul be able to be array
    #fwhm_vel = np.ones_like(spectra['freq']) * fwhm
    sigma = fwhm / (2 * (2 * np.log(2)) ** .5)
    # just doppler relation
    # the observed frequencies
    fwhm_freq = (fwhm/c.c * model['freq_rest']).decompose().to(u.GHz)
    # the observed frequencies
    freqs = spectra['freq_lsr']
    sigmas_freq = fwhm_freq / (2 * (2 * np.log(2)) ** .5)

    gauss = lambda x, pos, amplitude, sigma: amplitude * np.exp(-(x-pos)**2/(2 * sigma**2))

    syn_spectra = np.zeros_like(spectra['freq_lsr'])

    #TODO does it preserve the units(?)
    # from the model table, BUT from the sigmas_freq array
    n = 0
    while n<len(model):
        #for line,sigma_freq in zip(model,sigmas_freq):
        pos = model['freq_rest'].quantity[n]
        if model['blend'][n]:
            # since W_calc is in Jy km/s
            # need to divide by fwhm in km/s to get Jy
            amplitude = model['W_calc_blend'].quantity[n] / (sigma * (2 * np.pi)**.5 )
            # add the gaussian to the spectra
            syn_spectra += gauss(freqs, model['freq_rest'].quantity[n].value, amplitude, sigmas_freq[n].value)
            n += np.max(model['blend'][n])
        elif not model['blend'][n]:
            # since W_calc is in Jy km/s
            # need to divide by fwhm in km/s to get Jy
            amplitude = model['W_calc'].quantity[n] / (sigma * (2 * np.pi)**.5 )
            # add the gaussian to the spectra
            syn_spectra += gauss(freqs, model['freq_rest'].quantity[n].value, amplitude, sigmas_freq[n].value)
            n += 1
        else:
            raise(Exception('Could not run while loop.'))
        if model['blend'].any():
            spectra.meta['synthetic_spectra_info'] = 'Blends are summed and accounted for.'

    # Now calculate this for a region around each line
    line_freqs_array = []
    model_spectra_blend = []
    model_spectra_single = []
    spectra_obs_lsr = []
    n = 0
    while n < len(model):
        # create array of arrays, with frequencies for
        # # calculating Gaussians
        f_start = (spectra['freq_lsr'] >= (model['freq_rest'].quantity[n] - 4*fwhm_freq[n])).nonzero()[0][0]
        f_stop = (spectra['freq_lsr'] <= (model['freq_rest'].quantity[n] + 4*fwhm_freq[n])).nonzero()[0][-1]
        # get the blended spectrum for line
        model_spectra_blend.append( syn_spectra[f_start:f_stop] )
        spectra_obs_lsr.append( spectra[fluxcol].quantity[f_start:f_stop] )
        line_freqs_array.append( spectra['freq_lsr'].quantity[f_start:f_stop] )
        amplitude = model['W_calc'].quantity[n] / (sigma * (2 * np.pi) ** .5)
        syn_line = gauss(spectra['freq_lsr'].quantity[f_start:f_stop].value, model['freq_rest'].quantity[n].value, amplitude, sigmas_freq[n].value)
        model_spectra_single.append( syn_line )
        n += 1
    # create astropy table to put into large model table
    # with spectra around each line.
    syn_lines_table = Table(data=[line_freqs_array, model_spectra_single, model_spectra_blend, spectra_obs_lsr],
                            names=['freq_rest', 'spectra_single', 'spectra_blend', 'spectra_obs_lsr'])
    spectra[modelname] = syn_spectra
    #model[modelname] = syn_lines_table

    return spectra, syn_lines_table