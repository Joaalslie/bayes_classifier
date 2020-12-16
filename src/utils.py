import numpy as np


def split_dataset(data, labels, num_classes):
        """

        """
        split_data = [[] for _ in range(num_classes)]
        for datapoint, label in zip(data, labels):
            if label >= num_classes:
                raise Exception("number of labels in dataset doesn't " + \
                    "correlate with the number of classes!")

            split_data[int(label)].append(datapoint)

        for i in range(num_classes):
            split_data[i] = np.array(split_data[i])

        return split_data
