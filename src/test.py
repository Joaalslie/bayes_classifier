import unittest

import distributions as dists


class TestMultivariateNormal(unittest.TestCase):
    """

    """
    def test_pdf(self):
        """

        """
        pass

    def test_pdf_exception(self):
        """

        """
        dist = dists.MultivariateNormal()
        # Call on pdf before training to invoke exception
        self.assertRaises(Exception, dist.pdf, None)

    
    def test_log_pdf(self):
        """

        """
        pass
    
    def test_log_pdf_exception(self):
        """

        """
        dist = dists.MultivariateNormal()
        # Call on logarithmic pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)


class TestNormal(unittest.TestCase):
    """

    """
    def test_pdf(self):
        """

        """
        pass

    def test_pdf_exception(self):
        """

        """
        dist = dists.Normal()
        # Call on pdf before training to invoke exception
        self.assertRaises(Exception, dist.pdf, None)
    
    def test_log_pdf(self):
        """

        """
        pass
    
    def test_log_pdf_exception(self):
        """

        """
        dist = dists.Normal()
        # Call on logarithmic pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)


class TestPoisson(unittest.TestCase):
    """

    """
    def test_pdf(self):
        """

        """
        pass
    
    def test_pdf_exception(self):
        """

        """
        dist = dists.Poisson()
        # Call on pdf before training to invoke exception
        self.assertRaises(Exception, dist.pdf, None)
    
    def test_log_pdf(self):
        """

        """
        pass
    
    def test_log_pdf_exception(self):
        """

        """
        dist = dists.Poisson()
        # Call on logarithmic pdf before training to invoke exception
        self.assertRaises(Exception, dist.log_pdf, None)


if __name__ == '__main__':  
    unittest.main()
