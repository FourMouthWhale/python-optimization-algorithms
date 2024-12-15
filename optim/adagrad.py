import numpy as np

class Adagrad:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.G = None

    def update(self, w, grad_w):
        if self.G is None:
            self.G = np.zeros_like(grad_w)

        self.G += grad_w ** 2
        dx = grad_w / (np.sqrt(self.G) + self.eps)
        return w - dx