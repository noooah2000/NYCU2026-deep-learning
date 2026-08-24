import numpy as np

class SGD:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        for i, grad in enumerate(self.model.grads_list):
            self.model.params_list[i] -= grad * self.lr

class Adam:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.beta1_t = self.beta1
        self.beta2_t = self.beta2
        self.m = [np.zeros_like(p) for p in model.params_list]
        self.v = [np.zeros_like(p) for p in model.params_list]

    def step(self):
        for i, grad in enumerate(self.model.grads_list):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_head = self.m[i] / (1 - self.beta1_t)
            v_head = self.v[i] / (1 - self.beta2_t)
            self.model.params_list[i] -= self.lr * m_head / (np.sqrt(v_head) + 1e-5)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
