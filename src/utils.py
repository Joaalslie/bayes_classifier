import numpy as np


def split_dataset(data, labels, num_classes):
        """

        """
        split_data = [[] for _ in range(num_classes)]
        for datapoint, label in zip(data, labels):
            split_data[int(label)].append(datapoint)
        
        for i in range(num_classes):
            split_data[i] = np.array(split_data[i])

        return split_data
