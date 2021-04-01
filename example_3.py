import matplotlib.pyplot as plt
import scipy.special as sps
import numpy as np

from bayes_classifier import BayesClassifier
from distribution import Normal, Gamma


class Program():
    """
    Object responsible for running the program and controlling the overall
    program flow.
    """
    def __init__(self):
        np.random.seed(1)

        gamma = Gamma()
        gamma.set_estimators(2)

        self.classifier = BayesClassifier(2)
        self.classifier.add_class(Normal(), 0)
        self.classifier.add_class(gamma, 1)

        self.main()

    def main(self):
        """
        Overall main function of the program. Generates data, trains model
        and measure accuracy of the model.
        """
        x_train, y_train = self.generate_normal_data(600)
        x_test, y_test = self.generate_normal_data(300, plot=False)

        self.classifier.fit(x_train, y_train)
        acc = self.classifier.accuracy(x_test, y_test)
        print("accuracy: ", acc)

    def generate_normal_data(self, N, plot=True):
        """
        Generate N data samples of normal distributed data for 1 class and
        gamma distributed data from 1 class.

        param N: total number of samples to generate
        type N: int
        param plot: True if the dataset is to be plotted
        type plot: bool, optional
        return: tuple of dataset and corresponding labels
        rtype: tuple
        """
        mu = 23
        sigma = 5
        shape = 2
        scale = 2

        x1 = np.random.normal(mu, sigma, int(N / 3))
        x2 = np.random.gamma(scale, shape, int(N / 3))

        x = np.concatenate((x1, x2))
        y = np.concatenate((np.zeros(int(N / 3)), np.ones(int(N / 3))))

        if plot:
            plt.hist(x2, 50, density=True)
            plt.hist(x1, 50, density=True)
            plt.show()

        return x, y


if __name__ == "__main__":
    Program()
