import numpy as np


def split_dataset(data, labels, num_classes):
        """
        Split dataset into smaller sets based on corresponding labels.

        :param data: list of feature vectors (dataset)
        :param labels: list of labels corresponding to the data
        :param num_classes: number of classes in the dataset
        :returns: list of split datasets with index corresponding to label
        :raises Exception: number of labels isn't equal to actual labels
        """
        split_data = [[] for _ in range(num_classes)]
        for datapoint, label in zip(data, labels):
            if label >= num_classes:
                raise Exception("number of labels in dataset doesn't " + \
                    "correlate with the number of classes!")

            split_data[int(label)].append(datapoint)

        for i in range(num_classes):
            split_data[i] = np.array(split_data[i])

        return np.array(split_data)
