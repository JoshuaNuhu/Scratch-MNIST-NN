import numpy as np
def initilise_parameters():
    w1=np.random.randn(10,784)
    b1=np.zeros((10,1))
    w2=np.random.randn(10,10)
    b2=np.zeros((10,1))
    return w1,b1,w2,b2
def forward_pass(w1,b1,w2,b2,x):
    z1=np.dot(x,w1)+b1
    a1= relu(z1)
    z2= np.dot(a1,w2)+b2
    a2 = softmax(z2)
    return z1,a1,z2,a2
def softmax(z):
    return np.exp(z)/ np.sum(np.exp(z))
def compute_cost(m,y_hat,y):
    cost=np.sum(np.sqrt(y-y_hat))/m
    return cost
def sigmoid(z):
    a=1/(1-np.exp(-z))
    return a
def relu(z):
    return max(0,z)
def derv_relu(z):
    return z>0
def backprop(z1,a1,z2,a2,w2,x,y):
    m=y.size
    dz2= a2-y
    dw2=np.dot(dz2,a1.T)/m
    db2= np.sum(dz2,axis=1,keepdims=True)/m
    dz1=np.dot(w2.T,a1)*derv_relu(z1)
    dw1=np.dot(dz1,x.T)/m
    db1= np.sum(dz1,axis=1,keepdims=True)/m
    return dw1,db1,dw2,db2
def update_parameters(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1-=alpha*dw1
    w2-=alpha*dw2
    b1-=alpha*db1
    b2-=alpha*db2
    return w1,b1,w2,b2