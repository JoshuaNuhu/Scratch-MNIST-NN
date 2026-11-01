import numpy as np
def initilise_parameters():
    w1=np.random.randn(10,784)
    b1=np.zeros((10,1))
    w2=np.random.randn(10,10)
    b2=np.zeros((10,1))
    return w1,b1,w2,b2
def compute_cost(m,y_hat,y):
    cost=np.sum(np.sqrt(y-y_hat))/m
    return cost
def sigmoid(z):
    a=1/(1-np.exp(-z))
    return a
def relu(z):
    return max(0,z)