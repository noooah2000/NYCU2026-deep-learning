import numpy as np
from activations import Sigmoid

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, loss_fn, activation): 
        self.loss_fn = loss_fn
        self.act = activation
        self.out_act = Sigmoid()

        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size)

        self.b1 = np.random.randn(1, hidden_size)
        self.b2 = np.random.randn(1, hidden_size)
        self.b3 = np.random.randn(1, output_size)

        self.params_list = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
        self.grads_list = [0] * len(self.params_list)

    def forward(self, X):
        self.z1 = self.act.forward(X @ self.W1 + self.b1)
        self.z2 = self.act.forward(self.z1 @ self.W2 + self.b2)
        self.y_pred = self.out_act.forward(self.z2 @ self.W3 + self.b3)
        return self.y_pred

    def backward(self, X, y_true):
        dy_pred = self.loss_fn.derivative(self.y_pred, y_true)
        delta3 = dy_pred * self.out_act.derivative(self.y_pred)
        delta2 = (delta3 @ self.W3.T) * self.act.derivative(self.z2)
        delta1 = (delta2 @ self.W2.T) * self.act.derivative(self.z1)

        self.dW1 = X.T @ delta1
        self.dW2 = self.z1.T @ delta2
        self.dW3 = self.z2.T @ delta3
        self.db1 = np.sum(delta1, axis=0, keepdims=True)
        self.db2 = np.sum(delta2, axis=0, keepdims=True)
        self.db3 = np.sum(delta3, axis=0, keepdims=True)

        self.grads_list[:] = [self.dW1, self.dW2, self.dW3, self.db1, self.db2, self.db3]