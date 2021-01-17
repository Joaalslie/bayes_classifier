import unittest

from .gamma import Gamma


class TestGamma(unittest.TestCase):
    def test_pdf(self):
        pass

    def test_pdf_exception(self):
        dist = Gamma()
        # Call on pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)

    def test_log_pdf(self):
        pass

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
