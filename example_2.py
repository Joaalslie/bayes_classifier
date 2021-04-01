from matplotlib import pyplot as plt
import numpy as np

from bayes_classifier import BayesClassifier
from distribution import Normal


class Program():
    """
    Object responsible for running the program and controlling the overall
    program flow.
    """
    def __init__(self):
        np.random.seed(1)

        self.classifier = BayesClassifier(3)
        self.classifier.add_class(Normal(), 0)
        self.classifier.add_class(Normal(), 1)
        self.classifier.add_class(Normal(), 2)

        self.main()

    def main(self):
        """
        Overall main function of the program. Generates data, trains model
        and measure accuracy of the model.
        """
        x_train, y_train = self.generate_normal_data(60)
        x_test, y_test = self.generate_normal_data(30, plot=False)

        self.classifier.fit(x_train, y_train)
        acc = self.classifier.accuracy(x_test, y_test)
        print("accuracy: ", acc)

    def generate_normal_data(self, N, plot=True):
        """
        Generate N data samples of normal distributed data for 3 classes.

        param N: total number of samples to generate
        type N: int
        param plot: True if the dataset is to be plotted
        type plot: bool, optional
        return: tuple of dataset and corresponding labels
        rtype: tuple
        """
        mu1 = 2.5
        mu2 = 4.3
        mu3 = 6.2
        sigma1 = 0.3
        sigma2 = 0.5
        sigma3 = 0.9

        x1 = np.random.normal(mu1, sigma1, int(N / 3))
        x2 = np.random.normal(mu2, sigma2, int(N / 3))
        x3 = np.random.normal(mu3, sigma3, int(N / 3))

        x = np.concatenate((x1, x2, x3))
        y = np.concatenate((np.zeros(int(N / 3)), np.ones(int(N / 3)),
            np.full((int(N / 3)), 2)))

        if plot:
            plt.figure(1, figsize=(8, 8))
            plt.scatter(
                x1, np.zeros(x1.shape), s=120, facecolors='none',
                edgecolors='black', linewidth=3.0, label='Class 1'
            )
            plt.scatter(x2, np.zeros(x2.shape), s=120, facecolors='none',
                edgecolors='blue', linewidth=3.0, label='Class 2'
            )
            plt.scatter(x3, np.zeros(x3.shape), s=120, facecolors='none',
                edgecolors='red', linewidth=3.0, label='Class 3'
            )
            plt.legend()
            plt.show()

        return x, y


if __name__ == "__main__":
    Program()
