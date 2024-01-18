""" 
Module errors. Contains:
error_prop Calculates the error range caused by the uncertainty of the fit
    parameters. Covariances are taken into account.
cover_to_corr: Converts covariance matrix into correlation matrix.
"""

import numpy as np
from scipy import stats 

def err_ranges(x, func, params, sigma):
    """
    Calculate confidence interval.

    Parameters
    ----------
    x : ndarray
        Input values.
    func : function
        Function to fit the data.
    params : ndarray
        Fitted parameters.
    sigma : ndarray
        Standard deviation of the parameters.

    Returns
    -------
    lower_bound : ndarray
        Lower bound of the confidence interval.
    upper_bound : ndarray
        Upper bound of the confidence interval.

    """
    alpha = 0.05  # You can adjust the confidence level as needed

    dof = max(0, len(x) - len(params))  # Degrees of freedom
    t_value = np.abs(stats.t.ppf(alpha / 2, dof))

    lower_bound = func(x, *(params - t_value * sigma))
    upper_bound = func(x, *(params + t_value * sigma))

    return lower_bound, upper_bound



def deriv(x, func, parameter, ip):
    """
    Calculates numerical derivatives from function
    values at parameter +/- delta.  Parameter is the vector with parameter
    values. ip is the index of the parameter to derive the derivative.

    """

    # print("in", ip, parameter[ip])
    # create vector with zeros and insert delta value for relevant parameter
    # delta is calculated as a small fraction of the parameter value
    scale = 1e-6   # scale factor to calculate the derivative
    delta = np.zeros_like(parameter, dtype=float)
    val = scale * np.abs(parameter[ip])
    delta[ip] = val  #scale * np.abs(parameter[ip])
    
    diff = 0.5 * (func(x, *parameter+delta) - func(x, *parameter-delta))
    dfdx = diff / val

    return dfdx


def covar_to_corr(covar):
    """ Converts the covariance matrix into a correlation matrix """

    # extract variances from the diagonal and calculate std. dev.
    sigma = np.sqrt(np.diag(covar))
    # construct matrix containing the sigma values
    matrix = np.outer(sigma, sigma)
    # and divide by it
    corr = covar/matrix
    
    return corr
                       
