import unittest

from .normal import Normal


class TestNormal(unittest.TestCase):
    def test_pdf(self):
        pass

    def test_pdf_exception(self):
        dist = Normal()
        # Call on pdf before training to invoke exception
        self.assertRaises(Exception, dist.pdf, None)

    def test_log_pdf(self):
        pass

    def test_log_pdf_exception(self):
        dist = Normal()
        # Call on logarithmic pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)
