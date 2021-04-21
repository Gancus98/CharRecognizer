#
#   Garnczarek Dawid - Programowanie Zaawansowane 2 - 02.2021
#
##########################################################################################

import string
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random




random.seed(42)

EPOCH_NUMBER = 5
MODEL_NAME = "model.h5"
DATA_PATH = "data_28x28.csv"

nr_to_letter = {k: v.upper() for k, v in enumerate(list(string.ascii_lowercase))}
numbers_dict = {
    26: '1',
    27: '2',
    28: '3',
    29: '4',
    30: '5',
    31: '6',
    32: '7',
    33: '8',
    34: '9',
    35: '0',
    36: '=',
    37: '+',
    38: '-',
    39: '*',
    40: '/',
}
nr_to_letter.update(numbers_dict)


def heatmap2d(arr: np.ndarray, title):
    plt.title(title)
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()


def load_data():
    df = pd.read_csv(DATA_PATH)
    features = df.values[:, 1:]
    labels = df.values[:, 0]
    return features, labels


def prepare_train_data(features, labels):
    features = features.reshape(len(features), 28, 28)
    data_x = features / 255
    data_y = labels
    return data_x, data_y


def split_data_train_test(data_x, data_y, percent=0.1):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    total_number_of_samples = data_x.shape[0]

    index = 0
    while index < total_number_of_samples:
        rand = random.uniform(0, 1)
        if rand > percent:
            train_images.append(data_x[index])
            train_labels.append(data_y[index])
        else:
            test_images.append(data_x[index])
            test_labels.append(data_y[index])
        index += 1
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)


def generate_chart_statistics(history):
    epochs = range(EPOCH_NUMBER)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(epochs, history.history['loss'], label='train set')
    ax1.plot(epochs, history.history['val_loss'], label='test set')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(epochs, history.history['accuracy'], label='train set')
    ax2.plot(epochs, history.history['val_accuracy'], label='test set')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    plt.show()


features, labels = load_data()
data_x, data_y = prepare_train_data(features, labels)
train_images, train_labels, test_images, test_labels = split_data_train_test(data_x, data_y, 0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(41, activation='softmax')  # 26 -41
])

model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
history = model.fit(train_images, train_labels, epochs=EPOCH_NUMBER, validation_data=(test_images, test_labels))
model.save(MODEL_NAME)

generate_chart_statistics(history)
