import numpy as np
import math

from ..base_classes import Distribution


class MultivariateNormal(Distribution):
    """
    Represents the multivariate normal distribution.
    """
    def __init__(self):
        self.is_fit = False

        self.mu = None
        self.sigma = None

    def pdf(self, x):
        """
        Compute multivariate normal pdf based on given x value.

        :param x: parameter value for multivariate normal pdf
        :returns: the result of the multivariate normal pdf
        :raises Exception: Maximum likelihood estimators have not been set
        """
        if self.is_fit:
            size = len(x)
            det = np.linalg.det(self.sigma)
            norm_const = 1.0 / np.power((2*np.pi), size / 2) * \
                np.power(det, 0.5)
            x_mu = np.matrix(x - self.mu)
            inv = np.matrix(self.sigma).I
            result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
            return norm_const * result
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def log_pdf(self, x):
        """
        Compute multivariate normal log pdf based on given x value.

        :param x: parameter value for multivariate normal log pdf
        :returns: the result of the multivariate normal log pdf
        :raises Exception: Maximum likelihood estimators have not been set
        """
        if self.is_fit:
            size = len(x)
            x_mu = np.matrix(x - self.mu)
            inv = np.matrix(self.sigma).I
            log_det = np.linalg.slogdet(self.sigma)[1]
            val = ((size / 2) * (2 * np.pi)) \
                + log_det + ((x_mu * inv * x_mu.T))
            return -0.5 * val
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def fit(self, data):
        """
        Compute maximum likelihood estimators of the multivariate normal
        distribution.

        :param data: data used to compute estimators
        :returns: None
        """
        self.mu = np.mean(data, axis=0)
        self.sigma = np.cov(data.T)
        self.is_fit = True
