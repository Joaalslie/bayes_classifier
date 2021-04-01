import numpy as np
import unittest

from .poisson import Poisson


class TestPoisson(unittest.TestCase):
    def create_pdf(self, alpha):
        dist = Poisson()
        # Set the parameter manually for simplicity
        dist.alpha = alpha
        dist.is_fit = True
        return dist

    def test_pdf(self):
        dist = self.create_pdf(3)
        y = dist.pdf(1)
        # 2 Decimals should be sufficient
        assert np.round(y, 2) == 0.15

    def test_pdf_negative_value_exception(self):
        # Ensure that Exception is raised when x is less than 0
        dist = self.create_pdf(3)
        self.assertRaises(Exception, dist.pdf, -1)

    def test_pdf_estimator_exception(self):
        dist = Poisson()
        # Call on pdf before training to invoke exception
        self.assertRaises(Exception, dist.pdf, None)

    def test_log_pdf(self):
        dist = self.create_pdf(3)
        y = dist.log_pdf(1)
        # 2 Decimals should be sufficient
        assert np.round(y, 2) == -1.90

    def test_log_pdf_negative_value(self):
        # Ensure that Exception is raised when x is less than 0
        dist = self.create_pdf(3)
        self.assertRaises(Exception, dist.log_pdf, -1)

    def test_log_pdf_exception(self):
        dist = Poisson()
        # Call on logarithmic pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)
    
    def test_fit_alpha(self):
        dist = Poisson()
        data = np.array([-4, -3, -2, 2, 3, 4])
        dist.fit(data)
        assert dist.alpha == 0
    
    def test_fit_is_fit(self):
        dist = Poisson()
        data = np.array([0])
        dist.fit(data)
        assert dist.is_fit == True
