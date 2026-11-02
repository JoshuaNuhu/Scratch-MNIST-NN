import mnist
from init import gradient_descent
mnist.datasets_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

X_train = mnist.train_images()
y_train = mnist.train_labels()
X_test  = mnist.test_images()
y_test  = mnist.test_labels()

X_train = X_train.reshape(X_train.shape[0], -1)  
m, n = X_train.shape                               # m = 60000, n = 784
w1,b1,w2,b2= gradient_descent(X_train,y_train,100,0.1)