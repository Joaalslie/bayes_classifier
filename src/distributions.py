from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):
    """
        An abstract base class acting as parent-class for probability density
        functions. The main aim of the class is to ensure that classes
        inheriting it have implemented a pdf-method.
    """
    @abstractmethod
    def pdf(self):
        pass


class multivariateNormal(Distribution):
    """
        A class which represents the multivariate normal distribution.
    """
    def __init__(self):
        """

        """
        self.is_fit = False

        self.mu = None
        self.sigma = None

    def pdf(self, x):
        """

        """
        if self.is_fit:
            size = len(x)
            det = np.linalg.det(self.sigma)
            norm_const = 1.0 / (np.power((2*np.pi), size / 2) * np.power(det, 0.5))
            x_mu = np.matrix(x - self.mu)
            inv = np.matrix(self.sigma).I
            result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
            return norm_const * result
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def log_pdf(self):
        """

        """
        pass


class Normal(Distribution):
    """
        A class which represents the singlevariate normal distribution.
    """
    def __init__(self):
        """

        """
        pass

    def pdf(self):
        """

        """
        pass

    def log_pdf(self):
        """

        """
        pass
