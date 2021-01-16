import unittest

from .poisson import Poisson


class TestPoisson(unittest.TestCase):
    def test_pdf(self):
        pass

    def test_pdf_negative_value(self):
        pass

    def test_pdf_exception(self):
        dist = Poisson()
        # Call on pdf before training to invoke exception
        self.assertRaises(Exception, dist.pdf, None)

    def test_log_pdf(self):
        pass

    def test_log_pdf_negative_value(self):
        pass

    def test_log_pdf_exception(self):
        dist = Poisson()
        # Call on logarithmic pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)
