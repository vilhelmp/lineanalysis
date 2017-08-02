


import astropy.units as u
import astropy.constants as c
import numpy as np
from . import utils

def calculate_syn(linedata=None,
                  ntot=None,
                  tex=None,
                  source_size=None,
                  beam_size=None,
                  species=None,
                  fwhm=None,
                  usetau=True,
                  returnjy=True,
                  spectra=None,
                  fluxcol=None,
                  ilims=[None,None],
                  vsys=None,
                  modelname="model_flux_1",
                  dnu=0.001 * u.GHz,
                  verbose=True,
                  qrot_method='cdms',
                  use_ilims=True
                  ):

    # Step 0, sort table according to frequency
    linedata.sort('freq_rest')

    # Step 1, calculate integrated synthetic line fluxes
    # from Ntot and Tex for the given lines in input "model"
    # This function will calculate Jy if returnjy=True
    # and then use the input beam and source size to also
    # account for beam dilution. This is total integrated fluxes.
    linemodel = utils.calc_line_fluxes(linedata=linedata,
                                      ntot=ntot,
                                      tex=tex,
                                      source=source_size,
                                      beam=beam_size,
                                      species=species,
                                      fwhm=fwhm,
                                      usetau=True,
                                      ilims=ilims,
                                      returnjy=True,
                                      verbose=verbose,
                                      qrot_method=qrot_method,
                                      )
        
    # Step 2, integrate the observed spectrum around frequency for
    # each line. This step is not crucial, but for fitting Ntot or Tex
    # it is useful. However, be careful to compare the same intensity.
    # i.e. integrated vs. fitted Gaussian etc.
    linemodel = utils.integrate_all_lines(spectra=spectra,
                                             model=linemodel,
                                             linedata=linedata,
                                             fluxcol=fluxcol,
                                             ilims=ilims,
                                             vsys=vsys,
                                             verbose=verbose,
                                             )
    # Step 3, check the model for line blends

    linemodel = utils.check_for_blend(model=linemodel,
                                      dnu=dnu,
                                      verbose=verbose)

    if linemodel.meta['blends_present']:
        if verbose:
            print('Blends present, process blends.')
        # Step 4, calculate a summed integrated synthetic line flux
        linemodel = utils.process_model_blends(model=linemodel)
    # Step 5, calculate the synthetic spectrum and put it in
    # a new column named as the input parameter "modelname".
    # note that linemodel_syn is different from the linemodel
    # data structure.

    spectra_syn, linemodel_syn = utils.calc_synthetic_spectrum(model=linemodel,
                                                                  spectra=spectra,
                                                                  fluxcol=fluxcol,
                                                                  modelname=modelname,
                                                                  fwhm=fwhm,
                                                                  verbose=verbose,
                                                                  )

    linemodel.meta['ilims'] = ilims

    # add a column with velocity for each line
    # makes plotting some what easier
    try:
        linemodel_syn['vel_rest'] = [i.quantity.to(u.km/u.s, 
            equivalencies=u.doppler_radio(j)) for (i,j) in zip(
                linemodel_syn['freq_rest'],
                linemodel['freq_rest'].quantity)]
    except(AttributeError): # if theres only one entry
        if verbose:
            print('WARN:Only one line, ugly hack ahead. Check units.')
        linemodel_syn['vel_rest'] = [(i*u.GHz).to(u.km/u.s, 
            equivalencies=u.doppler_radio(j*u.GHz)) for (i,j) in zip(
                linemodel_syn['freq_rest'],
                linemodel['freq_rest'])]
    return spectra_syn, linemodel_syn, linemodel


def subtract_obs_calc(data=None,
                      minf=lambda y,obs,err: (y-obs)**2/err**2,
                      use_ilims=True,
                      npars=1.,
                      nobs=0,
                      verbose=False):
    #TODO clean up!!!
    """
    :param data: list of Astropy Tables
        with columns ['W_calc', 'W_calc_blend', 'blend', 'W_obs']
    :param minf: function to minimize
                1 pos: model
                2 pos: observations
                3 pos: uncertainty
    :return:
    """
    if use_ilims:
        calc_blend = 'W_calc_ilims_blend'
        calc = 'W_calc_ilims'
    else:
        calc_blend = 'W_calc_blend'
        calc = 'W_calc'
    results = np.array([])
    n = 0
    while n < len(data):
        if data['blend'][n]:
            # if it is a blend
            mdldiff = minf(data[calc_blend].quantity[n],
                           data['W_obs'].quantity[n],
                           data['W_obs_err'].quantity[n].to(
                               data['W_obs'].quantity[n].unit)
                           ).decompose()
            results = np.append(results, mdldiff)
            # now add the number of blends
            # to skip them
            # jump one extra! otherwise we will underfit!
            n += np.max(data[n]['blend'])+1
        else:
            # if not a blend
            mdldiff = minf(data[calc].quantity[n],
                           data['W_obs'].quantity[n],
                           data['W_obs_err'].quantity[n].to(
                               data['W_obs'].quantity[n].unit)
                           ).decompose()
            results = np.append(results, mdldiff)
            # just add one
            n += 1
    # the second value returned is the reduced Chi square
    # over all lines calculated
    if len(data)>1:
        chisq_red = np.sum(results)/float(len(results)-npars)
        chisq = np.sum(results)
    else:
        if verbose:
            print('Only one line, returning Chi square only.')
        chisq_red = np.sum(results)
        chisq = np.sum(results)
    return results, chisq_red, chisq





#class SyntheticModel(self):
#    @u.quantity_input(dnu=u.Hz)
#    def __init__(self):
#        return None
#    def spectrum(self, ntot, tex):
#        return None





def calculate_grid_ntot(linedata=None,
                      ntots=None,
                      tex=None,
                      source_size=None,
                      beam_size=None,
                      species=None,
                      fwhm=None,
                      usetau=True,
                      returnjy=True,
                      spectra=None,
                      fluxcol=None,
                      ilims=[None,None],
                      vsys=None,
                      dnu=0.001 * u.GHz,
                      minf=lambda y, obs, err: (y - obs)**2/err**2,
                      verbose=True,
                      qrot_method='func',
                      taupenalty=False,
                      use_ilims=True,
                      ):
    """
    :param linedata:
    :param ntot:
    :param tex:
    :param source_size:
    :param beam_size:
    :param species:
    :param fwhm:
    :param usetau:
    :param returnjy:
    :param spectra:
    :param fluxcol:
    :param ilims:
    :param vsys:
    :param dnu:
    :param minf:
    :return:
    """
    print('Remember to put in both line IDs if you have a blend.')
    if len(linedata)<2:
        print('Only one line, turn on verbose')
    models = []
    results = []
    for n in ntots:
        spectra_syn, linemodel_syn, linemodel = calculate_syn(linedata=linedata,
                      ntot=n,
                      tex=tex,
                      source_size=source_size,
                      beam_size=beam_size,
                      species=species,
                      fwhm=fwhm,
                      usetau=usetau,
                      returnjy=returnjy,
                      spectra=spectra,
                      fluxcol=fluxcol,
                      ilims=ilims,
                      vsys=vsys,
                      modelname="model_flux_1",
                      dnu=dnu,
                      verbose=verbose,
                      qrot_method=qrot_method,
                      use_ilims=use_ilims,
                      )
        if len(linedata)<2:
            #print(linemodel_syn.colnames)
            if verbose:
                print('Points used:{0}'.format(len(linemodel_syn['spectra_single'][0])) )
        models.append([spectra_syn, linemodel_syn, linemodel])
        iloss, chisq_red, chisq = subtract_obs_calc(data=linemodel,
                                              minf=minf,
                                              use_ilims = use_ilims,
                                              npars=1.,
                                              verbose=verbose)
        results.append([iloss.value, chisq_red.value, chisq.value])
    print('Done.')
    return models, results




def calculate_grid(linedata=None,
                      ntots=None,
                      texs=None,
                      source_size=None,
                      beam_size=None,
                      species=None,
                      fwhm=None,
                      usetau=True,
                      returnjy=True,
                      spectra=None,
                      fluxcol=None,
                      ilims=[None,None],
                      vsys=None,
                      dnu=0.001 * u.GHz,
                      minf=lambda y, obs, err: (y - obs)**2/err**2,
                      verbose=True,
                      use_ilims=True,
                      qrot_method='func'):
    spectra_syn = spectra.copy()
    models = np.zeros([len(texs), len(ntots)])
    results_chisq_red = np.zeros([len(texs), len(ntots)])
    results_chisq = np.zeros([len(texs), len(ntots)])
    for j in range(len(ntots)):
        for i in range(len(texs)):
            # here I use spectra_syn over and over again.
            # it will create a new table column for each synthetic spectrum
            spectra, linemodel_syn, linemodel = calculate_syn(linedata=linedata,
                          ntot=ntots[j],
                          tex=texs[i],
                          source_size=source_size,
                          beam_size=beam_size,
                          species=species,
                          fwhm=fwhm,
                          usetau=usetau,
                          returnjy=returnjy,
                          spectra=spectra_syn,
                          fluxcol=fluxcol,
                          ilims=ilims,
                          vsys=vsys,
                          modelname="model_flux_{0}_{1}".format(i,j),
                          dnu=dnu,
                          verbose=verbose,
                          use_ilims=use_ilims,
                          qrot_method=qrot_method,
                          )
            #imodels = np .append(imodels, [spectra_syn, linemodel_syn, linemodel])
            iloss, chisq_red, chisq = subtract_obs_calc(data=linemodel,
                                                  minf=minf,
                                                  use_ilims=use_ilims,
                                                  )
            #TODO: here save the iloss into matrix
            #models[i,j] = imodels
            results_chisq_red[i,j] = chisq_red
            results_chisq[i,j] = chisq.value
    print('Done.')
    return results_chisq_red, results_chisq, linemodel


def calculate_leastsq_ntot(linedata=None,
                      ntot0=None,
                      ntot_bounds= [None, None],
                      tex=None,
                      source_size=None,
                      beam_size=None,
                      species=None,
                      fwhm=None,
                      usetau=True,
                      returnjy=True,
                      spectra=None,
                      fluxcol=None,
                      ilims=[None,None],
                      vsys=None,
                      dnu=0.001 * u.GHz,
                      minf=lambda y, obs, err: (y - obs)**2/err**2,
                      verbose=False,
                      use_ilims=True,
                      qrot_method='func',
                      ):
    #from scipy import optimize
    # when using leastsq
    #minf = lambda y, obs, err: (y - obs)**2  # so its the least squares
    import lmfit

    args=dict(linedata = linedata,
        tex = tex,
        source_size = source_size,
        beam_size = beam_size,
        species = species,
        fwhm = fwhm,
        usetau = usetau,
        returnjy = returnjy,
        spectra = spectra,
        fluxcol = fluxcol,
        ilims = ilims,
        vsys = vsys,
        modelname="model_flux_{0}".format(tex),
        dnu = dnu,
        minf = minf,
        verbose = verbose,
        use_ilims = use_ilims,
                )

    #x0 = [ntot0] # initial guesses
    pars = lmfit.Parameters()
    if ntot_bounds[0] != None:
        pars.add('ntot', value=ntot0,
                           min = ntot_bounds[0],
                           max = ntot_bounds[1],
                           )
    else:
        if ntot_bounds[0] != None:
            pars.add(name='ntot', value=ntot0,
                                   )

    def fitfunc(pars, **args):
        ntot = pars['ntot'].value
        # unpack the arguments (in the correct order...)
        spectra, linemodel_syn, linemodel = calculate_syn(linedata=args['linedata'],
                                                      ntot=ntot *u.cm**-2,
                                                      tex=args['tex'],
                                                      source_size=args['source_size'],
                                                      beam_size=args['beam_size'],
                                                      species=args['species'],
                                                      fwhm=args['fwhm'],
                                                      usetau=args['usetau'],
                                                      returnjy=args['returnjy'],
                                                      spectra=args['spectra'],
                                                      fluxcol=args['fluxcol'],
                                                      ilims=args['ilims'],
                                                      vsys=args['vsys'],
                                                      modelname=args['modelname'],
                                                      dnu=args['dnu'],
                                                      verbose=args['verbose'],
                                                      use_ilims=args['use_ilims'],
                                                      qrot_method=qrot_method)

        iloss, chisq_red = subtract_obs_calc(data=linemodel,
                                         minf=args['minf'],
                                         use_ilims=args['use_ilims'],
                                         )

        return chisq_red

    #res = optimize.minimize(fitfunc, x0, args=args, method='nelder-mead',
    #         options={'xtol': 1e-8, 'disp': True})
    #res = optimize.leastsq(fitfunc, x0, args=args,
    #                     full_output=1, epsfcn=0.0001)

    mini = lmfit.Minimizer(fitfunc, pars, fcn_kws=args)

    # first solve with Nelder-Mead
    #print('First solve with Nelder-Mead')
    #out1 = mini.minimize(method='Nelder')

    #print('Then solve with LM using the solution from NM as start.')
    # then solve with Levenberg-Marquardt using the
    # Nelder-Mead solution as a starting point
    #res = mini.minimize(method='leastsq', params=out1.params)
    res = mini.minimize(method='leastsq')

    print(res)
    return res, mini




def calculate_leastsq(linedata=None,
                      ntot0=None,
                      ntot_bounds= [None, None],
                      tex0=None,
                      tex_bounds= [None, None],
                      source_size=None,
                      beam_size=None,
                      species=None,
                      fwhm=None,
                      usetau=True,
                      returnjy=True,
                      spectra=None,
                      fluxcol=None,
                      ilims=[None,None],
                      vsys=None,
                      dnu=0.001 * u.GHz,
                      minf=lambda y, obs, err: (y - obs)**2/err**2,
                      verbose=False,
                      use_ilims=True,
                           ):
    #from scipy import optimize
    # when using leastsq
    #minf = lambda y, obs, err: (y - obs)**2  # so its the least squares
    import lmfit

    args=dict(linedata = linedata,
        source_size = source_size,
        beam_size = beam_size,
        species = species,
        fwhm = fwhm,
        usetau = usetau,
        returnjy = returnjy,
        spectra = spectra,
        fluxcol = fluxcol,
        ilims = ilims,
        vsys = vsys,
        modelname="model_flux_leastsq_search",
        dnu = dnu,
        minf = minf,
        verbose = verbose,
        use_ilims = use_ilims,
                )

    #x0 = [ntot0] # initial guesses
    pars = lmfit.Parameters()
    if ntot_bounds[0] != None:
        pars.add('ntot', value=ntot0,
                           min = ntot_bounds[0],
                           max = ntot_bounds[1],
                           )
    else:
        if ntot_bounds[0] != None:
            pars.add(name='ntot', value=ntot0,
                                   )

    if tex_bounds[0] != None:
        pars.add('tex', value=tex0,
                           min = tex_bounds[0],
                           max = tex_bounds[1],
                           )
    else:
        if tex_bounds[0] != None:
            pars.add(name='tex', value=tex0,
                                   )

    def fitfunc(pars, **args):
        ntot = pars['ntot'].value
        tex = pars['tex'].value
        # unpack the arguments (in the correct order...)
        spectra, linemodel_syn, linemodel = calculate_syn(linedata=args['linedata'],
                                                      ntot=ntot *u.cm**-2,
                                                      tex=tex*u.K,
                                                      source_size=args['source_size'],
                                                      beam_size=args['beam_size'],
                                                      species=args['species'],
                                                      fwhm=args['fwhm'],
                                                      usetau=args['usetau'],
                                                      returnjy=args['returnjy'],
                                                      spectra=args['spectra'],
                                                      fluxcol=args['fluxcol'],
                                                      ilims=args['ilims'],
                                                      vsys=args['vsys'],
                                                      modelname=args['modelname'],
                                                      dnu=args['dnu'],
                                                      verbose=args['verbose'],
                                                      use_ilims=args['use_ilims'],
                                                      )

        iloss, chisq_red = subtract_obs_calc(data=linemodel,
                                         minf=args['minf'],
                                         use_ilims=args['use_ilims'],
                                         )

        return chisq_red

    # res = optimize.minimize(fitfunc, x0, args=args, method='nelder-mead',
    #         options={'xtol': 1e-8, 'disp': True})
    # res = optimize.leastsq(fitfunc, x0, args=args,
    #                     full_output=1, epsfcn=0.0001)

    mini = lmfit.Minimizer(fitfunc, pars, fcn_kws=args)

    # first solve with Nelder-Mead
    # print('First solve with Nelder-Mead')
    # out1 = mini.minimize(method='Nelder')

    # print('Then solve with LM using the solution from NM as start.')
    # then solve with Levenberg-Marquardt using the
    # Nelder-Mead solution as a starting point
    # res = mini.minimize(method='leastsq', params=out1.params)
    res = mini.minimize(method='differential_evolution')

    print(res)
    return res, mini




