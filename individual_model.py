"""
Indvidual representing a model class.
"""

import numpy as np
import random as rd
import copy
from evol_model import create_model, choose

class Individual_model():
    def __init__(self, fitness_function, chromosomes = None, name = "???"):

        if chromosomes is None:
            self.chromosomes = {
                0 : {
                    "activation" : choose("activation"),
                    "do_pool" : choose("do_pool"),
                    "do_skip" : choose("do_skip"),
                    "pooling_type" : choose("pooling_type"),

                    "n_filter_a" : choose("n_filter_a"),
                    "kernel_size_a" : choose("kernel_size_a"),
                    "do_BN_a" : choose("do_BN_a"),

                    "n_filter_b" : choose("n_filter_b"),
                    "kernel_size_b" : choose("kernel_size_b"),
                    "do_BN_b" : choose("do_BN_b"),
                },
            }
        else:
            self.chromosomes = chromosomes

        self.fitness_function = fitness_function
        self.fitness_score = 0
        self.name = name


    def mutate(self):
        
        block, chrr = rd.choice(list(self.chromosomes.items()))
        hp_key = rd.choice(list(chrr.keys()))
        hp_new_value = choose(hp_key)
        self.chromosomes[block][hp_key] = hp_new_value


    def crossover(self, individual):
        L_childrens = list()
        for _ in range(2):
            children = self.copy()
            children.fitness_score = min(self.fitness_score, individual.fitness_score)
            for block, chrr in individual.chromosomes.items():
                for hp_key, hp_value in chrr.items():
                    if rd.random() > .5:
                        children.chromosomes[block][hp_key] = hp_value
            L_childrens.append(children)
        return L_childrens

    def copy(self):
        indiv = Individual_model(self.fitness_function, chromosomes = copy.deepcopy(self.chromosomes), name = self.name)
        indiv.fitness_score = self.fitness_score
        return indiv

    def fit(self):
        acc = self.fitness_function(self.chromosomes)
        self.fitness_score = acc
        return acc