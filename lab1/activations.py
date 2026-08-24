import numpy as np

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)
    
class ReLU:
    def forward(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return (x > 0).astype(float)
    
class NoActivation:
    def forward(self, x):
        return x
        
    def derivative(self, x):
        return np.ones_like(x)
