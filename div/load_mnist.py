import tensorflow as tf 
import numpy as np
from utils import unison_shuffled_copies

def load_mnist(frac = 1):
    #Loading...
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    x_test = np.load("data/x_test.npy")
    y_test = np.load("data/y_test.npy")
    #Shuffle x and y the same way...
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_test, y_test = unison_shuffled_copies(x_test, y_test)
    
    return (x_train[:int(60000//frac)], y_train[:int(60000//frac)]), (x_test[:int(10000//frac)], y_test[:int(10000//frac)])

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    np.save("data/x_train.npy", x_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/x_test.npy", x_test)
    np.save("data/y_test.npy", y_test)

