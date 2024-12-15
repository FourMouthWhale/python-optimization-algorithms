import numpy as np

class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad_w):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(grad_w)
            self.v = np.zeros_like(grad_w)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_w ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 加入权重衰减项
        update = (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)) + self.weight_decay * w
        return w - update