


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
                  ):

    # Step 0, sort table according to frequency
    linedata.sort('freq_rest')

    # Step 1, calculate integrated synthetic line fluxes
    # from Ntot and Tex for the given lines in input "model"
    # This function will calculate Jy if returnjy=True
    # and then use the input beam and source size to also
    # account for beam dilution.
    linemodel = utils.calc_line_fluxes(linedata=linedata,
                                      ntot=ntot,
                                      tex=tex,
                                      source=source_size,
                                      beam=beam_size,
                                      species=species,
                                      fwhm=fwhm,
                                      usetau=True,
                                      returnjy=True,
                                      verbose=verbose,
                                      qrot_method='cdms',
                                      )

    # Step 2, integrate the observed spectrum around frequency for
    # each line.
    # This step is not crucial, but for fitting Ntot or Tex
    # it is useful.
    linemodel = utils.integrate_all_lines(spectra=spectra,
                                             model=linemodel,
                                             linedata=linedata,
                                             fluxcol=fluxcol,
                                             ilims=ilims,
                                             vsys=vsys,
                                             verbose=verbose,
                                             )

    # Step 3, check the model for line blends
    linemodel = utils.check_for_blend(model=linemodel, dnu=0.001 * u.GHz,
                                      verbose=verbose,
                                      )

    if linemodel['blend'].any():
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


    return spectra_syn, linemodel_syn, linemodel


def subtract_obs_calc(data=None,
                      minf=lambda y,obs,err:(y-obs)**2,
                      taupenalty=False):
    """
    :param data: list of Astropy Tables
        with columns ['W_calc', 'W_calc_blend', 'blend', 'W_obs']
    :param minf: function to minimize
                1 pos: model
                2 pos: observations
                3 pos: uncertainty
    :param taupenalty: adds the tau value to the minfunction results.
            this way a penalty is added to the value, so that optically thick solutions are worse.
    :return:
    """
    results = np.array([])
    n = 0
    while n < len(data):
        if data['blend'][n]:
            # if it is a blend
            mdldiff = minf(data['W_calc_blend'][n],data['W_obs'][n], 0) + int(taupenalty)*data['tau_calc_blend'][n]
            results = np.append(results, mdldiff)
            # now add the number of blends
            # to skip them
            n += np.max(data[n]['blend'])
        else:
            # if not a blend
            mdldiff = minf(data['W_calc'][n],data['W_obs'][n], 0) + int(taupenalty)*data['tau_calc'][n]
            results = np.append(results, mdldiff)
            # just add one
            n += 1

    return results, np.mean(results)


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
                      minf=lambda y, obs, err: (y - obs) ** 2,
                      verbose=True,
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
                      )

        models.append([spectra_syn, linemodel_syn, linemodel])
        iloss, iloss_mean = subtract_obs_calc(data=linemodel, minf=minf)
        results.append([iloss, iloss_mean])
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
                      minf=lambda y, obs, err: (y - obs) ** 2,
                      verbose=True,
                      taupenalty=False):
    spectra_syn = spectra.copy()
    models = np.zeros([len(texs), len(ntots)])
    results = np.zeros([len(texs), len(ntots)])
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
                          )
            #imodels = np .append(imodels, [spectra_syn, linemodel_syn, linemodel])
            iloss, iloss_mean = subtract_obs_calc(data=linemodel, minf=minf, taupenalty=taupenalty)
            #TODO: here save the iloss into matrix
            #models[i,j] = imodels
            results[i,j] = iloss_mean
    print('Done.')
    return results, linemodel

