import numpy as np
import math

from .. import ManualEstimatorDistribution


class Gamma(ManualEstimatorDistribution):
    """
    Represents the singlevariate Gamma distribution
    """
    def __init__(self):
        self.is_fit = False
        # Gamma function: assume alpha is a positive integer
        self.gamma = lambda alpha: math.factorial(alpha - 1)

        self.alpha = None
        self.beta = None

    def pdf(self, x):
        """
        Compute singlevariate gamma pdf based on given x value.

        :param x: parameter value for gamma pdf
        :returns: the result of the gamma pdf
        :raises Exception: Maximum likelihood estimators have not been set
        """
        if self.is_fit:
            return (1 / np.power(self.beta, self.alpha) * \
                self.gamma(self.alpha)) * \
                np.power(x, self.alpha - 1) * \
                np.power(np.e, -x / self.beta)
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def log_pdf(self, x):
        """
        Compute singlevariate gamma log pdf based on given x value.

        :param x: parameter value for gamma log pdf
        :returns: the result of the gamma log pdf
        :raises Exception: Maximum likelihood estimators have not been set
        """
        if self.is_fit:
            return (self.alpha - 1) * np.log(x) - (x / self.beta) - \
                np.log(self.gamma(self.alpha)) + \
                (self.alpha * (np.log(self.beta)))
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def fit(self, x):
        """
        Compute maximum likelihood estimators of the gamma distribution.

        :param data: data used to compute estimators
        :returns: None
        :raises Exception: alpha is not a positive integer!
        """
        if not isinstance(self.alpha, int) and self.alpha < 0:
            raise Exception("Alpha needs to be set as a positive integer!")

        self.beta = np.mean(x) / self.alpha
        self.is_fit = True

    def set_estimators(self, *estimators):
        """
        Set the maximum likelihood estimator alpha manually.

        :param *estimators: list of estimator values (contains only alpha)
        :returns: None
        """
        self.alpha = estimators[0]
