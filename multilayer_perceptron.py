import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from matplotlib import pyplot

TOP_WORDS = 5000    # Only keep the 5000 most frequently used words and zero out the rest
MAX_WORDS = 500     # Create a limit for the number of words in a review. This will truncate longer reviews
                    # and add zero-padding to shorter reviews

def read_in_dataset():
    (training_data, training_labels), (test_data, test_labels) = imdb.load_data(num_words=TOP_WORDS)

    training_data = sequence.pad_sequences(training_data, maxlen=MAX_WORDS)
    test_data = sequence.pad_sequences(test_data, maxlen=MAX_WORDS)

    return {
        "training_data": training_data,
        "training_labels": training_labels,
        "test_data": test_data,
        "test_labels": test_labels
    }

def create_model():
    model = Sequential()
    model.add(Embedding(TOP_WORDS, 32, input_length=MAX_WORDS))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    datasets = read_in_dataset()
    model = create_model()
    model.fit(datasets["training_data"], datasets["training_labels"], validation_data=(datasets["test_data"], datasets["test_labels"]), epochs=2, batch_size=128, verbose=2)
    results = model.evaluate(datasets["test_data"], datasets["test_labels"], verbose=0)
    print("Final accuracy: {}%".format(results[1] * 100))


if __name__ == '__main__':
    main()