import mnist
import numpy as np
from init import gradient_descent,forward_pass,prediction
mnist.datasets_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

# X_train = mnist.train_images()
# y_train = mnist.train_labels()
# X_test  = mnist.test_images()
# y_test  = mnist.test_labels()

# X_train = X_train.reshape(X_train.shape[0], -1)  
# m, n = X_train.shape                               # m = 60000, n = 784
# w1,b1,w2,b2= gradient_descent(X_train,y_train,100,0.1)

X_train = mnist.train_images().reshape(-1, 784).T / 255.0   # (784, 60000)
y_train = mnist.train_labels()
X_test  = mnist.test_images().reshape(-1, 784).T / 255.0
y_test  = mnist.test_labels()

w1, b1, w2, b2 = gradient_descent(X_train, y_train, 500, 0.1)

*_, a2 = forward_pass(w1, b1, w2, b2, X_test)
print("Test accuracy:", np.mean(prediction(a2) == y_test))