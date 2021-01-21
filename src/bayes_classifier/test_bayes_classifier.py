import unittest

from bayes_classifier import BayesClassifier
from distribution import Distribution, Normal


class TestBayesClassifier(unittest.TestCase):
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

    def test_remove_class(self):
        pass

    def test_remove_class_exception(self):
        # Ensure that exception is raised when trying to add distribution
        # on an invalid idx in the remove_class() function.
        pass

    def test_add_class(self):
        pass

    def test_invalid_idx_add_class(self):
        # Ensure that exception is raised when trying to add distribution
        # on an invalid idx in the add_class() function.
        pass

    def test_invalid_dist_add_class(self):
        # Ensure that exception is raised when trying to add invalid
        # class/distribution in the add_class() function.
        pass

    def test_predict(self):
        pass

    def test_predict_exception(self):
        # Ensure that exception is raised when trying to predict before
        # training the model.
        pass

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
