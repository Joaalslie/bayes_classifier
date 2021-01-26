import numpy as np
import unittest

from bayes_classifier import BayesClassifier
from distribution import Distribution, Normal, MultivariateNormal


class TestBayesClassifier(unittest.TestCase):
    np.random.seed(1)

    def create_model(self):
        # Create model
        classifier = BayesClassifier(2)
        classifier.add_class(MultivariateNormal(), 0)
        classifier.add_class(MultivariateNormal(), 1)
        # Create data
        mu1 = np.array([1.0, 1.0])
        mu2 = np.array([4.0, 4.0])
        sigma = np.array([[0.2, 0.0], [0.0, 0.2]])
        x1 = np.random.multivariate_normal(mu1, sigma, 3)
        x2 = np.random.multivariate_normal(mu2, sigma, 3)
        x = np.concatenate((x1, x2))
        y = np.array([0, 0, 0, 1, 1, 1])
        # Train and return model
        classifier.fit(x, y)
        return classifier

    def test_num_classes(self):
        # Ensure that num_classes is set upon construction
        classifier = BayesClassifier(2)
        assert classifier.num_classes == 2

    def test_set_log_pdf(self):
        # Ensure that set_log_pdf() function sets use_log_pdf attribute to
        # False
        classifier = BayesClassifier(2)
        classifier.set_log_pdf()
        assert classifier.use_log_pdf == True

    def test_unset_log_pdf(self):
        # Ensure that set_log_pdf() function sets use_log_pdf attribute to
        # True
        classifier = BayesClassifier(2)
        classifier.unset_log_pdf()
        assert classifier.use_log_pdf == False

    def test_add_class(self):
        dist = Normal()
        classifier = BayesClassifier(3)
        classifier.add_class(dist, 0)
        assert classifier.distributions[0] == dist

    def test_add_class_counter(self):
        classifier = BayesClassifier(5)
        classifier.add_class(Normal(), 0)
        assert classifier.added_classes == 1

    def test_negative_idx_add_class(self):
        # Ensure that exception is raised when trying to add distribution
        # on a negative idx in the add_class() function.
        classifier = BayesClassifier(7)
        self.assertRaises(Exception, classifier.add_class, Normal(), -1)
    
    def test_exceeding_idx_add_class(self):
        # Ensure that exception is raised when trying to add distribution
        # on an exceeding idx in the add_class() function.
        classifier = BayesClassifier(4)
        self.assertRaises(Exception, classifier.add_class, Normal(), 5)

    def test_invalid_dist_add_class(self):
        # Ensure that exception is raised when trying to add invalid
        # class/distribution in the add_class() function.
        classifier = BayesClassifier(2)
        self.assertRaises(Exception, classifier.add_class, 1, 1)

    def test_remove_class(self):
        # Ensure that distribution is removed from list after remove_class()
        # function has been called
        classifier = BayesClassifier(2)
        classifier.add_class(Normal(), 1)
        classifier.remove_class(1)
        assert classifier.distributions[1] == None
    
    def test_remove_class_counter(self):
        # Ensure that counter of added classes is correct after remove_class()
        # function has been called
        classifier = BayesClassifier(2)
        classifier.add_class(Normal(), 1)
        classifier.remove_class(1)
        assert classifier.added_classes == 0

    def test_remove_class_exception_negative_value(self):
        # Ensure that exception is raised when trying to remove class
        # on negative invalid idx in the remove_class() function.
        classifier = BayesClassifier(2)
        self.assertRaises(Exception, classifier.remove_class, -1)

    def test_remove_class_exception_exceeding_idx(self):
        # Ensure that exception is raised when trying to remove class
        # on negative invalid idx in the remove_class() function.
        classifier = BayesClassifier(2)
        self.assertRaises(Exception, classifier.remove_class, 2)

    def test_predict(self):
        # Ensure that the predict method works when using normal pdf
        classifier = self.create_model()
        mu = np.array([1.0, 1.0])
        sigma = np.array([[0.2, 0.0], [0.0, 0.2]])
        x = np.random.multivariate_normal(mu, sigma, 1)
        prediction = classifier.predict(x)
        assert prediction == 0

    def test_predict_log_pdf(self):
        # Ensure that predict method works when using log pdf
        classifier = self.create_model()
        classifier.set_log_pdf()
        mu = np.array([1.0, 1.0])
        sigma = np.array([[0.2, 0.0], [0.0, 0.2]])
        x = np.random.multivariate_normal(mu, sigma, 1)
        prediction = classifier.predict(x)
        assert prediction == 0

    def test_predict_exception(self):
        # Ensure that exception is raised when trying to predict before
        # training the model.
        classifier = BayesClassifier(2)
        assert classifier.is_fit != True
        self.assertRaises(Exception, classifier.predict, 1)

    def test_accuracy(self):
        pass

    def test_accuracy_exception(self):
        # Ensure that exception is raised when trying to measure the accuracy
        # before training the model.
        pass

    def test_fit(self):
        pass

    def test_fit_exception(self):
        # Ensure that exception is raised when trying to train the model
        # without adding all distributions/classes
        pass
