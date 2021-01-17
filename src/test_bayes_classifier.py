import unittest

from .bayes_classifier import BayesClassifier


class TestBayesClassifier(unittest.TestCase):
    def test_set_log_pdf(self):
        pass

    def test_unset_log_pdf(self):
        pass

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
