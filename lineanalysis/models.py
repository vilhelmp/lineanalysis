


import astropy.units as u
import astropy.constants as c
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
                  dnu=0.001 * u.GHz
                  ):

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
                                      )

    # Step 2, integrate the observed spectrum around frequency for
    # each line.
    # This step is not crucial, but for fitting Ntot or Tex
    # it is useful.
    linemodel = la.utils.integrate_all_lines(spectra=spectra,
                                             model=linemodel,
                                             linedata=linedata,
                                             fluxcol=fluxcol,
                                             ilims=ilims,
                                             vsys=vsys,
                                             )

    # Step 3, check the model for line blends
    linemodel = la.utils.check_for_blend(model=linemodel, dnu=0.001 * u.GHz)

    # Step 4, calculate a summed integrated synthetic line flux
    linemodel = la.utils.process_model_blends(model=linemodel)

    # Step 5, calculate the synthetic spectrum and put it in
    # a new column named as the input parameter "modelname".
    # note that linemodel_syn is different from the linemodel
    # data structure.
    spectra_syn, linemodel_syn = la.utils.calc_synthetic_spectrum(model=linemodel,
                                                                  spectra=spectra,
                                                                  fluxcol=fluxcol,
                                                                  modelname=modelname,
                                                                  fwhm=fwhm)


    return spectra_syn, linemodel_syn, linemodel