import numpy as np
from keras.datasets import imdb
from matplotlib import pyplot

def read_in_dataset():
    (training_data, training_labels), (test_data, test_labels) = imdb.load_data()
    return {
        "training_data": training_data,
        "training_labels": training_labels,
        "test_data": test_data,
        "test_labels": test_labels
    }

def main():
    datasets = read_in_dataset()


if __name__ == '__main__':
    main()