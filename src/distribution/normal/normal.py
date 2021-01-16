import numpy as np

from .. import SingleVariateDistribution


class Normal(SingleVariateDistribution):
    """
    Represents the singlevariate normal distribution.
    """
    def __init__(self):
        self.is_fit = False

        self.mu = None
        self.sigma = None

    def pdf(self, x):
        """
        Compute singlevariate normal pdf based on given x value.

        :param x: parameter value for singlevariate normal pdf
        :returns: the result of the singlevariate normal pdf
        :raises Exception: Maximum likelihood estimators have not been set
        """
        if self.is_fit:
            norm = 1 / np.sqrt(2 * np.pi * self.sigma)
            exp = np.e**(-(((x - self.mu)**2) / (2 * self.sigma)))
            return norm * exp
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def log_pdf(self, x):
        """
        Compute singlevariate normal log pdf based on given x value.

        :param x: parameter value for singlevariate normal pdf
        :returns: the result of the singlevariate normal log pdf
        :raises Exception: Maximum likelihood estimators have not been set
        """
        if self.is_fit:
            norm = np.log(1 / (np.sqrt(2 * np.pi) * self.sigma))
            exp = np.power(x - mu, 2) / (2 * np.power(self.sigma, 2))
            return norm - exp
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def fit(self, data):
        """
        Compute maximum likelihood estimators of the singlevariate normal 
        distribution.

        :param data: data used to compute estimators
        :returns: None
        """
        self.mu = np.mean(data)
        self.sigma = np.var(data)
        self.is_fit = True
