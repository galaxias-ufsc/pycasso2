import numpy as np


def continuum(x, y, returns='ratio', degr=6, niterate=5,
              lower_threshold=2, upper_threshold=3, verbose=False):
    """
    Builds a polynomial continuum from segments of a spectrum,
    given in the form of wavelength and flux arrays.

    Parameters
    ----------
    x : array-like
        Independent variable
    y : array-like
        y = f(x)
    returns : string
        Specifies what will be returned by the function
        'ratio' = ratio between fitted continuum and the spectrum
        'difference' = difference between fitted continuum and the
            spectrum
        'function' = continuum function evaluated at x
    degr : integer
        Degree of polynomial for the fit
    niterate : integer
        Number of rejection iterations
    lower_threshold : float
        Lower threshold for point rejection in units of standard
        deviation of the residuals
    upper_threshold : float
        Upper threshold for point rejection in units of standard
        deviation of the residuals
    verbose : boolean
        Prints information about the fitting

    Returns
    -------
    c : tuple
        c[0] : numpy.ndarray
            Input x coordinates
        c[1] : numpy.ndarray
            See parameter "returns".
    """

    xfull = copy.deepcopy(x)
    s = copy.deepcopy(y)
    m = np.ones(s.shape, dtype='bool')

    def f(m):
        return np.polyval(np.polyfit(x[m], s[m], deg=degr), x)

    for i in range(niterate):

        if len(x) == 0:
            print('Stopped at iteration: {:d}.'.format(i))
            break
        sig = np.std(s - f(m))[m]
        res = s - f(m)
        m = (res < upper_threshold * sig) & (res > -lower_threshold * sig)
        x = x[m]
        s = s[m]

    npoints = np.sum(m)

    if verbose:
        print('Final number of points used in the fit: {:d}'
              .format(np.sum(m)))
        print('Rejection ratio: {:.2f}'
              .format(1. - float(npoints)) / float(len(xfull))))

    p = np.polyfit(x[m], s[m], deg=degr)

    if returns == 'ratio':
        return xfull, y / np.polyval(p, xfull)

    if returns == 'difference':
        return xfull, y - np.polyval(p, xfull)

    if returns == 'function':
        return xfull, np.polyval(p, xfull)
