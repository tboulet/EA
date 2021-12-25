'''
Run the genetic algorithm.
'''

import time
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
kl = tf.keras.layers

from fitness import fitness
from AlgoGen import AlgoGen, best_score
from individual_model import Individual_model
from utils import *

#Population :
N = 20
#Generations :
G = 1000

ga = AlgoGen(
    Individual_model, 
    pop_size=N,
    fitness_function= lambda x : fitness(x, frac = 10),
    mutation_rate=0.4,
    crossover_rate=0.4,
    elitism = True,
    )

L_best_acc = list()
L_mean_acc = list()
L_worst_acc = list()

for _ in range(G):
    ga.step()
    accs = list(indiv.fitness_score for indiv in ga.population)
    L_best_acc.append(accs[-1])
    L_mean_acc.append(moy(accs))
    L_worst_acc.append(accs[0])

    plt.plot(L_best_acc, label = "Best")
    plt.plot(L_mean_acc, label = "Mean")
    plt.plot(L_worst_acc, label = "Worst")
    # plt.ylim((0,1))
    plt.legend()
    plt.savefig("figures/accs")
    plt.close()

print("End.")
