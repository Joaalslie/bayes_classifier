import numpy as np
import unittest

from .normal import Normal


class TestNormal(unittest.TestCase):
    def create_pdf(self, mu, sigma):
        dist = Normal()
        # Set the parameters manually for simplicity
        dist.mu = mu
        dist.sigma = sigma
        dist.is_fit = True
        return dist

    def test_pdf(self):
        dist = self.create_pdf(0, 1)
        y = dist.pdf(1)
        # 2 decimals should be sufficient
        assert np.round(y, 2) == 0.24

    def test_pdf_negative_x(self):
        dist = self.create_pdf(0, 1)
        y = dist.pdf(-1)
        assert np.round(y, 2) == 0.24

    def test_pdf_exception(self):
        dist = Normal()
        # Call on pdf before training to invoke exception
        self.assertRaises(Exception, dist.pdf, None)

    def test_log_pdf(self):
        dist = self.create_pdf(0, 1)
        y = dist.log_pdf(1)
        assert np.round(y, 2) == -1.42
    
    def test_log_negative(self):
        dist = self.create_pdf(0, 1)
        y = dist.log_pdf(-1)
        assert np.round(y, 2) == -1.42

    def test_log_pdf_exception(self):
        dist = Normal()
        # Call on logarithmic pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)

    def test_fit_mu(self):
        # Ensure that mean is correctly set after fit
        dist = Normal()
        data = np.array([-4, -3, -2, 2, 3, 4])
        dist.fit(data)
        assert dist.mu == 0
    
    def test_fit_sigma(self):
        # Ensure that variance is correctly set after fit
        dist = Normal()
        data = np.array([-5, -4, -3, -2, 2, 3, 4, 5])
        dist.fit(data)
        assert dist.sigma == 13.5
    
    def test_fit_is_fit(self):
        # Ensure that is_fit bool is correctly set after fit
        dist = Normal()
        dist.fit(np.array([0]))
        assert dist.is_fit == True
