import numpy as np

class RProp:
    def __init__(self, lr_plus=1.2, lr_minus=0.5, delta_max=50, delta_min=1e-6, initial_delta=0.1):
        self.lr_plus = lr_plus
        self.lr_minus = lr_minus
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.initial_delta = initial_delta
        self.delta = None
        self.sign_grad_prev = None

    def update(self, w, grad_w):
        if self.delta is None:
            self.delta = np.ones_like(grad_w) * self.initial_delta
            self.sign_grad_prev = np.sign(grad_w)

        sign_grad = np.sign(grad_w)
        self.delta *= (sign_grad == self.sign_grad_prev) * self.lr_plus + (sign_grad!= self.sign_grad_prev) * self.lr_minus
        self.delta = np.minimum(self.delta_max, np.maximum(self.delta_min, self.delta))
        self.sign_grad_prev = sign_grad

        dx = -self.delta * sign_grad
        return w + dx