import numpy as np

from pdf import PDF


class BayesClassifier():
    """

    """
    def __init__(self, num_classes):
        """

        """
        self.num_classes = num_classes
        self.added_classes = 0
        self.is_fit = False

        self.distributions = [[] for _ in range(num_classes)]

    def fit(self):
        """

        """
        pass

    def predict(self):
        """

        """
        if self.is_fit:
            predictions = []
            # Iterate over each distribution and make prediction
            for distribution in self.distributions:
                prediction = distribution.predict()
                predictions.append(prediction)
            
            # Convert prediction list to numpy array
            predictions = np.array(predictions)
            # Return the class index with the biggest value
            return np.argmax(predictions)
        else:
            raise Exception("Model has not been trained yet!")
