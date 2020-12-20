import numpy as np

from distributions import Distribution
from utils import split_dataset


class BayesClassifier():
    """

    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.added_classes = 0
        self.is_fit = False

        self.distributions = [[] for _ in range(num_classes)]
        self.prior_probabilities = np.empty(num_classes)

    def fit(self, data, labels):
        """

        """
        if self.added_classes != self.num_classes:
            raise Exception("Model is not ready to train!" + \ 
                "All classes needs to be added!")

        total_size = len(data)
        # Separate dataset into smaller datasets by label
        split_data = split_dataset(data, labels, self.num_classes)
        for i, distribution in enumerate(split_data):
            self.distributions[i].fit(split_data.T)
            # Compute prior probability for the class
            prior_prob = len(split_data) / total_size
            self.prior_probabilities[i] = prior_prob

        self.is_fit = True

    def predict(self, datapoint):
        """

        """
        if self.is_fit:
            predictions = []
            # Iterate over each distribution and make prediction
            for i, distribution in enumerate(self.distributions):
                prior_prob = self.prior_probabilities[i]
                prediction = distribution.pdf(datapoint)
                predictions.append(prediction * prior_prob)
            
            # Convert prediction list to numpy array
            predictions = np.array(predictions)
            # Return the class index with the biggest value
            return np.argmax(predictions)
        else:
            raise Exception("Model has not been trained yet!")

    def add_class(self, distribution, idx):
        """

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

        """
        # Check if idx exceeds list size (number of classes)
        if not (idx < 0) or (idx > len(self.distributions) - 1):
            self.distributions[idx] = None
            self.added_classes -= 1
        else:
            raise Exception("idx value is not supported")
