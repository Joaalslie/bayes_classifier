from abc import ABC, abstractmethod
import numpy as np
import math


class Distribution(ABC):
    """
        An abstract base class acting as parent-class for probability density
        functions. The main aim of the class is to ensure that classes
        inheriting it have implemented a pdf-method.
    """
    @abstractmethod
    def pdf(self, x):
        pass

    @abstractmethod
    def log_pdf(self, x):
        pass

    @abstractmethod
    def fit(self, x):
        pass


class SingleVariateDistribution(Distribution):
    """
        Parent class for single-variate distributions.
    """
    def plot_pdf(self, x_min, x_max, n=100):
        """

        """
        x = np.linspace(x_min, x_max, n)
        y = [self.pdf(datapoint) for datapoint in x]
        plt.plot(x, y)
        plt.show()
    
    def plot_log_pdf(self, x_min, x_max, n=100):
        """

        """
        x = np.linspace(start, stop, n)
        y = [self.log_pdf(datapoint) for datapoint in x]
        plt.plot(x, y)
        plt.show()


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

        """
        self.mu = np.mean(data, axis=0)
        self.sigma = np.cov(data.T)
        self.is_fit = True


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

        """
        if self.is_fit:
            norm = 1 / np.sqrt(2 * np.pi * self.sigma)
            exp = np.e**(-(((x - self.mu)**2) / (2 * self.sigma)))
            return norm * exp
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def log_pdf(self, x):
        """

        """
        if self.is_fit:
            norm = np.log(1 / (np.sqrt(2 * np.pi) * self.sigma))
            exp = np.power(x - mu, 2) / (2 * np.power(self.sigma, 2))
            return norm - exp
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def fit(self, data):
        """

        """
        self.mu = np.mean(data)
        self.sigma = np.var(data)
        self.is_fit = True


class Poisson(SingleVariateDistribution):
    """
        Represents the singlevariate Poisson distribution
    """
    def __init__(self):
        self.is_fit = False
        self.alpha = 0

    def pdf(self, x):
        """

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

        """
        self.alpha = np.mean(x)
        self.is_fit = True


class Gamma(SingleVariateDistribution):
    """
        Represents the singlevariate Gamma distribution
    """
    def __init__(self):
        self.is_fit = False

    def pdf(self, x):
        """

        """
        if self.is_fit:
            pass
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def log_pdf(self, x):
        """

        """
        if self.is_fit:
            pass
        else:
            raise Exception("Distribution doesn't have all parameters set!")

    def fit(self, x):
        """

        """
        self.is_fit = True
