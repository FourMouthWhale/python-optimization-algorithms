import numpy as np

class Adadelta:
    def __init__(self, rho=0.95, eps=1e-6):
        self.rho = rho
        self.eps = eps
        self.Eg2 = None
        self.Edx2 = None
        self.delta_x = None

    def update(self, w, grad_w):
        if self.Eg2 is None:
            self.Eg2 = np.zeros_like(grad_w)
            self.Edx2 = np.zeros_like(w)
            self.delta_x = np.zeros_like(w)

        self.Eg2 = self.rho * self.Eg2 + (1 - self.rho) * grad_w ** 2
        dx = np.sqrt((self.Edx2 + self.eps) / (self.Eg2 + self.eps)) * grad_w
        self.Edx2 = self.rho * self.Edx2 + (1 - self.rho) * dx ** 2
        self.delta_x = -dx
        return w + self.delta_x