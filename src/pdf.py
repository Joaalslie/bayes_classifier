from abc import ABC, abstractmethod
import numpy as np


class PDF(ABC):
    """
        An abstract base class acting as parent-class for probability density
        functions. The main aim of the class is to ensure that classes
        inheriting it have implemented a pdf-method.
    """
    @abstractmethod
    def pdf(self):
        pass


class multivariateNormal(PDF):
    """
        A class which represents the multivariate normal distribution.
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


class Normal(PDF):
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
