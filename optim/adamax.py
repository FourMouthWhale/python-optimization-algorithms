import numpy as np

class Adamax:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.u = None
        self.t = 0

    def update(self, w, grad_w):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(grad_w)
            self.u = np.zeros_like(grad_w)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_w
        self.u = np.maximum(self.beta2 * self.u, np.abs(grad_w))

        m_hat = self.m / (1 - self.beta1 ** self.t)
        dx = self.lr * m_hat / (self.u + self.eps)
        return w - dx