import numpy as np
def initilise_parameters():
    pass
def compute_cost(m,y_hat,y):
    cost=np.sum(np.sqrt(y-y_hat))/m
    return cost
def sigmoid(z):
    a=1/(1-np.exp(-z))
    return a
def relu(z):
    return max(0,z)