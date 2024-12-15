import numpy as np

class SparseAdam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad_w):
        self.t += 1
        if self.m is None:
            self.m = {}
            self.v = {}

        for index in np.ndindex(grad_w.shape):
            if index not in self.m:
                self.m[index] = np.zeros(1)
                self.v[index] = np.zeros(1)

            g = grad_w[index]
            self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * g
            self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * g ** 2

            m_hat = self.m[index] / (1 - self.beta1 ** self.t)
            v_hat = self.v[index] / (1 - self.beta2 ** self.t)

            dx = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            w[index] -= dx

        return w