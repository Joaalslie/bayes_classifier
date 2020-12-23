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
        pass


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
        pass


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
        pass


if __name__ == '__main__':  
    unittest.main()
