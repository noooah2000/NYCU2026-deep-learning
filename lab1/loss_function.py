import numpy as np

class MSE:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) /len(y_true)

class BCE:
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def derivative(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return ((y_pred - y_true) / (y_pred * (1 - y_pred))) / len(y_true)