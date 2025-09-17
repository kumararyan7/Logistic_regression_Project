import numpy as np 

def sigmoid (z):
    return 1/ (1+np.exp(-z))

def train(X,y,lr = 0.1,epochs=1000):

    m,n= X.shape
    weights = np.zeroes(n)
    bias =0

    for _ in range (epochs):
        linear = np.dot(X,weights)+bias
        y_pred = sigmoid(linear)

        dw= (1/m)* np.dot(X.T,(y_pred -y))
        db = (1/m)* np.sum(y_pred -y )

        weights -= lr *dw
        bias -= lr* db

    return weights,bias

def predict(X,weights,bias):
    return (sigmoid(np.dot(X,weights)+bias) >= 0.5).astype(int)
