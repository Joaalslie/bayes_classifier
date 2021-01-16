import numpy as np

from .. import SingleVariateDistribution


class Poisson(SingleVariateDistribution):
    """
    Represents the singlevariate Poisson distribution
    """
    def __init__(self):
        self.is_fit = False
        self.alpha = 0

    def pdf(self, x):
        """
        Compute singlevariate poisson pdf based on given x value.

        :param x: parameter value for poisson pdf
        :returns: the result of the poisson pdf
        :raises Exception: Maximum likelihood estimators have not been set
        """
        if self.is_fit:
            if x >= 0:
                enu = np.power(self.alpha, x) * np.power(np.e, -self.alpha)
                den = np.math.factorial(x)
                return num / den
            else:
                return 0
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def log_pdf(self, x):
        """
        Compute singlevariate poisson log pdf based on given x value.

        :param x: parameter value for poisson log pdf
        :returns: the result of the poisson log pdf
        :raises Exception: Maximum likelihood estimators have not been set
        """
        if self.is_fit:
            if x >= 0:
                return -np.log(np.math.factorial(x)) + \
                    x * np.log(self.alpha) - self.alpha
            else:
                return 0
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def fit(self, x):
        """
        Compute maximum likelihood estimators of the poisson distribution.

        :param data: data used to compute estimators
        :returns: None
        """
        self.alpha = np.mean(x)
        self.is_fit = True
