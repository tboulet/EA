import tensorflow as tf
import numpy as np
from evol_model import create_model

from div.load_mnist import load_mnist



def fitness(chromosomes, epochs = 5, frac = 1):
    """
    Return a measure of how a model is able to learn well during 5 epochs. This will be the function to maximize.
    The model should not have already been trained and has to be a "baby" model. Though, for balancing the important demand in number of epochs for the deepest and largest networks,
    you may use weights of previous great models in the first layers of deepest models (transfer learning).

    frac is the times you divide the dataset after shuffling (less data for less computation time)
    """
    
    # #Test fitness func
    # s = 0
    # for value in chromosomes[0].values():
    #     if type(value) in (float, int):
    #         s += value
    # return s

    (x_train, y_train), (x_test, y_test) = load_mnist(frac = frac)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    acc = tf.keras.metrics.Accuracy()
    model = create_model(chromosomes)

    model.compile(optimizer = tf.keras.optimizers.Adam(1e-3), loss = tf.losses.SparseCategoricalCrossentropy(), metrics = ["accuracy"])
    model.fit(x = x_train, y = y_train, epochs = epochs, verbose = 1)
    y_pred = model.predict(x_test)
    acc.update_state(np.argmax(y_pred, axis = -1), y_test)
    return acc.result().numpy()