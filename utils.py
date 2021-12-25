import random as rd
import sys 
import numpy as np

categories = ["n_filter", "kernel_size", "activation", "pooling_type", "do_skip", "n_layers"]

def stride(n_in, n_out, k, p):
    return int(n_in/n_out)

def moy(L):
    return sum(L)/len(L)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]