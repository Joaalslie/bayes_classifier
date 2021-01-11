import numpy as np

from distributions import Distribution
from utils import split_dataset


class BayesClassifier():
    """
    Represents a Bayes Classifier ML Model.

    :param num_classes: number of classes in the model
    """
    added_classes = 0
    is_fit = False
    # Decides if the classifier should use the log-pdf of each distribution
    # when classifying
    use_log_pdf = False

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.distributions = [[] for _ in range(num_classes)]
        self.prior_probabilities = np.empty(num_classes)

    def fit(self, data, labels):
        """
        Train the model based on the given data and labels.

        :param data: dataset containing feature vectors to train the model on
        :param labels: labels corresponding to the data provided
        :returns: None
        :raises Exception: All classes has not been added.
        """
        if self.added_classes != self.num_classes:
            raise Exception("Model is not ready to train!" + 
                "All classes needs to be added!")

        total_size = len(data)
        # Separate dataset into smaller datasets by label
        split_data = split_dataset(data, labels, self.num_classes)
        for i, subset in enumerate(split_data):
            self.distributions[i].fit(subset)
            # Compute prior probability for the class
            prior_prob = len(split_data) / total_size
            self.prior_probabilities[i] = prior_prob

        self.is_fit = True

    def predict(self, datapoint):
        """
        Make a class prediction of a single datapoint.

        :param datapoint: vector to make prediction of
        :returns: Index of predicted class in list of classes
        :raises Exception: model needs to be trained before prediction
        """
        if self.is_fit:
            predictions = []
            # Iterate over each distribution and make prediction
            for i, distribution in enumerate(self.distributions):
                prior_prob = self.prior_probabilities[i]
                if self.use_log_pdf:
                    prediction = distribution.log_pdf(datapoint)
                else:
                    prediction = distribution.pdf(datapoint)
                
                predictions.append(prediction * prior_prob)
            
            # Convert prediction list to numpy array
            predictions = np.array(predictions)
            # Return the class index with the biggest value
            return np.argmax(predictions)
        else:
            raise Exception("Model has not been trained yet!")

    def accuracy(self, data, labels):
        """
        Classify data and compute the accuracy based on the labels provided.

        :param data: dataset to compute accuracy of
        :param labels: corresponding labels for the given dataset
        :returns: accuracy of the Bayes Classifier
        """
        length = len(labels)
        correct_predictions = 0
        for datapoint, label in zip(data, labels):
            prediction = self.predict(datapoint)
            if prediction == label:
                correct_predictions += 1

        return correct_predictions / length

    def add_class(self, distribution, idx):
        """
        Add a new class to the list of classes.

        :param distribution: distribution object of the given class
        :param idx: index location for the new class
        :returns: None
        :raises Exception: index or distribution is not supported
        """
        if isinstance(distribution, Distribution):
            # Check if idx exceeds list size (number of classes)
            if not (idx < 0) or (idx > len(self.distributions) - 1):
                self.distributions[idx] = distribution
                self.added_classes += 1
            else:
                raise Exception("idx value is not supported")
        else:
            raise Exception("Distribution is not supported!")

    def remove_class(self, idx):
        """
        Remove a class from the list of classes based on the given index.

        :param idx: index of the class to remove from the distributions list
        :returns: None
        :raises Exception: idx param is either to small or large for the list
        """
        # Check if idx exceeds list size (number of classes)
        if not (idx < 0) or (idx > len(self.distributions) - 1):
            self.distributions[idx] = None
            self.added_classes -= 1
        else:
            raise Exception("idx value is not supported")

    def set_log_pdf(self):
        """
        Ensure that the logarithmic probability density function of each
        estimated distribution is used when classifying.
        """
        self.use_log_pdf = True

    def unset_log_pdf(self):
        """
        Ensure that the probability density function of each estimated
        distribution is use when classifying.
        """
        self.use_log_pdf = False
