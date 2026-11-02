# from sklearn.metrics import accuracy_score
# import numpy as np
# def initilise_parameters():
#     w1=np.random.randn(10,784)
#     b1=np.zeros((10,1))
#     w2=np.random.randn(10,10)
#     b2=np.zeros((10,1))
#     return w1,b1,w2,b2
# def forward_pass(w1,b1,w2,b2,x):
#     z1=np.dot(x,w1)+b1
#     a1= relu(z1)
#     z2= np.dot(a1,w2)+b2
#     a2 = softmax(z2)
#     return z1,a1,z2,a2
# def softmax(z):
#     return np.exp(z)/ np.sum(np.exp(z))
# def compute_cost(y_hat,y):
#     m=y.size
#     cost=np.sum(np.sqrt(y-y_hat))/m
#     return cost
# def sigmoid(z):
#     a=1/(1-np.exp(-z))
#     return a
# def relu(z):
#     return max(0,z)
# def derv_relu(z):
#     return z>0
# def backprop(z1,a1,z2,a2,w2,x,y):
#     m=y.size
#     dz2= a2-y
#     dw2=np.dot(dz2,a1.T)/m
#     db2= np.sum(dz2,axis=1,keepdims=True)/m
#     dz1=np.dot(w2.T,a1)*derv_relu(z1)
#     dw1=np.dot(dz1,x.T)/m
#     db1= np.sum(dz1,axis=1,keepdims=True)/m
#     return dw1,db1,dw2,db2
# def update_parameters(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
#     w1-=alpha*dw1
#     w2-=alpha*dw2
#     b1-=alpha*db1
#     b2-=alpha*db2
#     return w1,b1,w2,b2
# def prediction(a2):
#     return np.argmax(a2,0)

# def gradient_descent(x,y,iteration, alpha):
#     w1,b1,w2,b2= initilise_parameters()
#     for i in range(iteration):
#         z1,a1,z2,a2=forward_pass(w1,b1,w2,b2,x)
#         dw1,db1,dw2,db2=backprop(z1,a1,z2,a2,w2,x,y)
#         w1,b1,w2,b2=update_parameters(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha)
#         if i %10 == 0:
#             print("iteration",i)
#             print("Cost:",compute_cost(prediction(a2),y))
#             print("Accuracy:",accuracy_score(y,prediction(a2)))
#     return w1,b1,w2,b2
            
import numpy as np

def init_params():
    w1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def relu(z):
    return np.maximum(0, z)

def derv_relu(z):
    return (z > 0).astype(float)

def softmax(z):
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

def forward_pass(w1, b1, w2, b2, x):
    z1 = w1 @ x + b1
    a1 = relu(z1)
    z2 = w2 @ a1 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    oh = np.zeros((10, y.size))
    oh[y, np.arange(y.size)] = 1
    return oh

def compute_cost(a2, y_oh):
    m = y_oh.shape[1]
    return -np.sum(y_oh * np.log(a2 + 1e-9)) / m

def backprop(z1, a1, z2, a2, w2, x, y_oh):
    m = y_oh.shape[1]
    dz2 = a2 - y_oh
    dw2 = dz2 @ a1.T / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = (w2.T @ dz2) * derv_relu(z1)
    dw1 = dz1 @ x.T / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    return dw1, db1, dw2, db2

def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    return w1 - alpha * dw1, b1 - alpha * db1, w2 - alpha * dw2, b2 - alpha * db2

def prediction(a2):
    return np.argmax(a2, axis=0)

def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    y_oh = one_hot(y)
    for i in range(iterations):
        z1, a1, z2, a2 = forward_pass(w1, b1, w2, b2, x)
        grads = backprop(z1, a1, z2, a2, w2, x, y_oh)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, *grads, alpha)
        if i % 10 == 0:
            print(f"iteration {i} | cost {compute_cost(a2, y_oh):.4f} | acc {np.mean(prediction(a2) == y):.4f}")
    return w1, b1, w2, b2