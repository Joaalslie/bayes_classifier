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
        self.assertRaises(Exception, dist.pdf, -3)

    def test_pdf_exception(self):
        dist = Gamma()
        # Call on pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)

    def test_log_pdf(self):
        dist = self.create_pdf(2, 2)
        y = dist.log_pdf(1)
        # 2 decimals should be sufficient
        assert np.round(y, 2) == -1.89
    
    def test_log_pdf_negative_x(self):
        dist = self.create_pdf(2, 2)
        self.assertRaises(Exception, dist.log_pdf, -3)

    def test_log_pdf_exception(self):
        dist = Gamma()
        # Call on logarithmic pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)

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

    def test_fit_beta(self):
        # Ensure that mean is correctly set after fit
        dist = Gamma()
        dist.alpha = 2
        data = np.array([-4, -3, -2, 2, 3, 4])
        dist.fit(data)
        assert dist.beta == 0

    def test_fit_exception(self):
        # Ensure that exception is raised when fit() is called and alpha is
        # not set yet
        dist = Gamma()
        data = np.array([-4, -3, -2, 2, 3, 4])
        self.assertRaises(Exception, dist.fit, data)

    def test_set_estimators(self):
        # Ensure that alpha is correctly set after fit
        dist = Gamma()
        dist.set_estimators(2)
        assert dist.alpha == 2

    def test_set_estimators_not_is_fit(self):
        # Ensure that is_fit is not True when only alpha has been set
        dist = Gamma()
        dist.set_estimators(2)
        assert dist.is_fit == False

    def test_is_fit_set_estimators_and_fit(self):
        # Ensure that is_fit is True when both alpha and beta has been set
        dist = Gamma()
        data = np.array([-4, -3, -2, 2, 3, 4])
        dist.set_estimators(2)
        dist.fit(data)
        assert dist.is_fit == True
