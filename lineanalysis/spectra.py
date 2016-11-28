




"""
Beam Dilution and Beam Filling are the same thing.
bd = O_s/O_b
    O_s=source solid angle
    O_b = beam solid angle
Ntot \propto W/bd
Nup = \propto W/bd
i.e.
W \propto Ntot * bd
W \propto Nup * bd

"""


def cutout_lines(linedata=None,
                 spectrum=None,
                 ):

    """
        CUTOUT from plotting.py
        vel_lims = linedata['vel'].quantity[num] + np.array([-2, 2]) * lwidth * u.km / u.s + vsys
        velmin, velmax = vel_lims[0].value - linedata['vel'][num], vel_lims[1].value - linedata['vel'][num]
        vel_mask = (spectra['vel'] > vel_lims[0]) * (spectra['vel'] < vel_lims[1])
        # add this to function --- END
        xvalues = spectra['vel'][vel_mask] - linedata['vel'][num]
        yvalues = spectra[spos][vel_mask]
    """
    return None


def fill_between_steps(x, y1, y2=0, h_align='mid', ax=None, **kwargs):
    ''' Fills a hole in matplotlib: Fill_between for step plots.

    Parameters :
    ------------

    x : array-like
        Array/vector of index values. These are assumed to be equally-spaced.
        If not, the result will probably look weird...
    y1 : array-like
        Array/vector of values to be filled under.
    y2 : array-Like
        Array/vector or bottom values for filled area. Default is 0.

    **kwargs will be passed to the matplotlib fill_between() function.

    '''
    # If no Axes opject given, grab the current one:
    if ax is None:
        ax = plt.gca()
    # First, duplicate the x values
    xx = x.repeat(2)[1:]
    # Now: the average x binwidth
    xstep = (x[1:] - x[:-1]).mean()
    # Now: add one step at end of row.
    xx = sp.append(xx, xx.max() + xstep)

    # Make it possible to change step alignment.
    if h_align == 'mid':
        xx -= xstep / 2.
    elif h_align == 'right':
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)
    if type(y2) == sp.ndarray:
        y2 = y2.repeat(2)

    # now to the plotting part:
    return ax.fill_between(xx, y1, y2=y2, **kwargs)
