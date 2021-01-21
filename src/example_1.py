from matplotlib import pyplot as plt
import numpy as np

from bayes_classifier import BayesClassifier
from distribution import MultivariateNormal


class Program():
    """
    Object responsible for running the program and controlling the overall
    program flow.
    """
    def __init__(self):
        self.classifier = BayesClassifier(2)
        self.classifier.add_class(MultivariateNormal(), 0)
        self.classifier.add_class(MultivariateNormal(), 1)

        self.main()
    
    def main(self):
        """
        Overall main function of the program. Generates data, trains model
        and measure accuracy of the model.
        """
        x_train, y_train = self.generate_multivariate_data(100, plot=True)
        x_test, y_test = self.generate_multivariate_data(50, plot=True)

        self.classifier.fit(x_train, y_train)
        acc = self.classifier.accuracy(x_test, y_test)
        print("accuracy: ", acc)

    def generate_multivariate_data(self, N, plot=True):
        """
        Generate N multivariate gaussian samples

        param N: total number of samples to generate
        type N: int

        param plot: True if the dataset is to be plotted
        type plot: bool, optional

        return: tuple of dataset and corresponding labels
        rtype: tuple
        """
        mu1 = np.array([1, 1])
        mu2 = np.array([2.0, 2.0])
        sigma = np.array([[0.2, 0.0], [0.0, 0.2]])

        x1 = np.random.multivariate_normal(mu1, sigma, int(N / 2))
        x2 = np.random.multivariate_normal(mu2, sigma, int(N / 2))
        x = np.concatenate((x1, x2))
        y = np.concatenate((np.zeros(int(N / 2)), np.ones(int(N / 2))))

        if plot:
            plt.figure(1, figsize=(8, 8))
            plt.scatter(
                x1[:, 0], x1[:, 1], s=120, facecolors='none',
                edgecolors='black', linewidth=3.0, label='Class 1'
            )
            plt.scatter(x2[:, 0], x2[:, 1], s=120, facecolors='none',
                edgecolors='blue', linewidth=3.0, label='Class 2'
            )
            plt.legend()
            plt.show()

        return x, y
        

if __name__ == "__main__":
    Program()
