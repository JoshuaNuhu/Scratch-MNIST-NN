import mnist

mnist.datasets_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

X_train = mnist.train_images()
y_train = mnist.train_labels()
X_test  = mnist.test_images()
y_test  = mnist.test_labels()

print(X_train.shape, X_test.shape)