import numpy as np

class RMSProp:
    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.Eg2 = None

    def update(self, w, grad_w):
        if self.Eg2 is None:
            self.Eg2 = np.zeros_like(grad_w)

        self.Eg2 = self.rho * self.Eg2 + (1 - self.rho) * grad_w ** 2
        dx = self.lr * grad_w / (np.sqrt(self.Eg2) + self.eps)
        return w - dx