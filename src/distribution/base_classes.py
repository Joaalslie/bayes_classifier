from abc import ABC, abstractmethod
import numpy as np


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

    def foo(self):
        print("it works!")


class SingleVariateDistribution(Distribution):
    """
    Parent class for single-variate distributions.
    """
    def plot_pdf(self, x_min, x_max, n=100):
        """
        Create a plot of the pdf of the underlying distribution.

        :param x_min: minimum x value for plotting
        :param x_max: maximum x value for plotting
        :param n: number of datapoints used for plot (standard set to 100) 
        :returns: None
        """
        x = np.linspace(x_min, x_max, n)
        y = [self.pdf(datapoint) for datapoint in x]
        plt.plot(x, y)
        plt.show()

    def plot_log_pdf(self, x_min, x_max, n=100):
        """
        Create a plot of the log pdf of the underlying distribution.

        :param x_min: minimum x value for plotting
        :param x_max: maximum x value for plotting
        :param n: number of datapoints used for plot (standard set to 100) 
        :returns: None
        """
        x = np.linspace(start, stop, n)
        y = [self.log_pdf(datapoint) for datapoint in x]
        plt.plot(x, y)
        plt.show()


class ManualEstimatorDistribution(SingleVariateDistribution):
    """
    Parent class for distribution which require one or more maximum
    likelihood estimators to be set manually.
    """
    @abstractmethod
    def set_estimators(self, *estimators):
        pass
