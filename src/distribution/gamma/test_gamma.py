import numpy as np
import unittest

from .gamma import Gamma


class TestGamma(unittest.TestCase):
    def create_pdf(self, alpha, beta):
        dist = Gamma()
        # Set the parameters manually for simplicity
        dist.alpha = alpha
        dist.beta = beta
        dist.is_fit = True
        return dist

    def test_pdf(self):
        dist = self.create_pdf(2, 2)
        y = dist.pdf(1)
        # 2 decimals should be sufficient
        assert np.round(y, 2) == 0.15
    
    def test_pdf_negative_x(self):
        dist = self.create_pdf(2, 2)
        y = dist.pdf(1)
        # 2 decimals should be sufficient
        assert np.round(y, 2) == 0.15

    def test_pdf_exception(self):
        dist = Gamma()
        # Call on pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)

    def test_log_pdf(self):
        dist = self.create_pdf(2, 2)
        y = dist.log_pdf(1)
        # 2 decimals should be sufficient
        assert np.round(y, 2) == -1.89

    def test_log_pdf_exception(self):
        dist = Gamma()
        # Call on logarithmic pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)

    def test_set_estimators(self):
        pass

    def test_set_estimators_exception_negative_int(self):
        dist = Gamma()
        # Ensure that exception is raised when using negative integer as
        # parameter in the set_estimators() function
        self.assertRaises(Exception, dist.set_estimators, -1)
    
    def test_set_estimators_exception_non_int(self):
        dist = Gamma()
        # Ensure that exception is raised when using negative integer as
        # parameter in the set_estimators() function
        self.assertRaises(Exception, dist.set_estimators, 2.34)
