import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod


class GradientDescent(ABC):
    def __init__(self, init_point, eps, max_iter):
        self.eps = eps
        self.max_iter = max_iter
        self.w = init_point
        self.iter = 0

    @abstractmethod
    def step(self):
        pass

    def optimize(self):
        prev_w = 2 * self.eps + self.w
        while np.linalg.norm(prev_w - self.w) > self.eps and self.iter < self.max_iter:
            self.iter += 1
            prev_w = self.w
            self.step()


class ClassicalMethod(GradientDescent):
    def __init__(self, grad_f, init_point, rate=0.01, eps=1e-6, max_iter=10000):
        super(ClassicalMethod, self).__init__(init_point, eps, max_iter)
        self.rate = rate
        self.grad_f = grad_f

    def step(self):
        self.w = self.w - self.rate * self.grad_f(self.w)


class Momentum(GradientDescent):
    def __init__(self, grad_f, init_point, rate=0.01, eps=1e-6, max_iter=10000, lamb=0.1):
        super(Momentum, self).__init__(init_point, eps, max_iter)
        self.grad_f = grad_f
        self.rate = rate
        self.v = np.zeros(*init_point.shape)
        self.lamb = lamb
        self.gamma = 1 - self.lamb
        self.eta = (1 - self.gamma) * self.rate

    def step(self):
        self.v = self.gamma * self.v + self.eta * self.grad_f(self.w)
        self.w = self.w - self.v


class NAG(GradientDescent):
    def __init__(self, grad_f, init_point, rate=0.01, eps=1e-6, max_iter=10000, lamb=0.1):
        super(NAG, self).__init__(init_point, eps, max_iter)
        self.grad_f = grad_f
        self.rate = rate
        self.v = np.zeros(*init_point.shape)
        self.lamb = lamb
        self.gamma = 1 - self.lamb
        self.eta = (1 - self.gamma) * self.rate

    def step(self):
        self.v = self.gamma * self.v + self.eta * self.grad_f(self.w - self.gamma * self.v)
        self.w = self.w - self.v


class RMSprop(GradientDescent):
    def __init__(self, grad_f, init_point, eps=1e-6, max_iter=10000, lamb=0.1):
        super(RMSprop, self).__init__(init_point, eps, max_iter)
        self.grad_f = grad_f
        self.gamma = 1 - lamb
        self.g = np.random.rand(*init_point.shape)

    def step(self):
        self.g = self.gamma * self.g + (1 - self.gamma) * self.grad_f(self.w) * self.grad_f(self.w)
        self.w = self.w - (1 - self.gamma) * self.grad_f(self.w) / np.sqrt(self.g + self.eps)


class AdaDelta(GradientDescent):
    def __init__(self, grad_f, init_point, alpha=0.9, eps=1e-6, max_iter=10000):
        super(AdaDelta, self).__init__(init_point, eps, max_iter)
        self.grad_f = grad_f
        self.alpha = alpha
        self.g = np.random.rand(*init_point.shape)
        self.delta_u = np.ones(*init_point.shape)*0.01


    def step(self):
        self.g = self.alpha * self.g + (1 - self.alpha) * self.grad_f(self.w) * self.grad_f(self.w)
        delta_l = self.grad_f(self.w) * np.sqrt(self.delta_u + self.eps)/np.sqrt(self.g + self.eps)
        self.delta_u = self.alpha * self.delta_u + (1 - self.alpha) * delta_l * delta_l
        self.w = self.w - delta_l


class Adam(GradientDescent):
    def __init__(self, grad_f, init_point, rate = 0.01, alpha=0.999, eps=1e-6, max_iter=10000, lamb=0.1):
        super(Adam, self).__init__(init_point, eps, max_iter)
        self.alpha = alpha
        self.rate = rate
        self.grad_f = grad_f
        self.lamb = lamb
        self.gamma = 1 - self.lamb
        self.V = np.zeros(*init_point.shape)
        self.G = np.zeros(*init_point.shape)
        self.v = 0
        self.g = 0

    def step(self):
        self.V = self.gamma * self.V + (1 - self.gamma) * self.grad_f(self.w)
        self.G = self.alpha * self.G + (1 - self.alpha) * self.grad_f(self.w)**2
        self.v = self.V / (1 - self.gamma**(self.iter+1))
        self.g = self.G / (1 - self.alpha**(self.iter+1))
        self.w = self.w - self.rate * self.v / (np.sqrt(self.g) + self.eps)

def grad(w):
    return np.array([4*w[0]-4, 4*w[1]-8])
    # return np.array([2*w[0] + 64*(w[0] + 0.75)**3, 8*w[1] - 2.4])
    # return np.array([64*w[0]**3 + 4*w[0], 8*w[1] - 8])


optimizers = {
    "Классический метод": ClassicalMethod,
    "Momentum": Momentum,
    "NAG": NAG,
    "RMSprop": RMSprop,
    "AdaDelta": AdaDelta,
    "Adam": Adam
}

statistics = []
for name, cls in optimizers.items():
    micro_statistics = []
    for i in range(100):
        initial_point = np.random.rand(2) * 10
        optimizer = cls(grad, initial_point)
        optimizer.optimize()
        print(name, ": ", initial_point, optimizer.w, ", ", optimizer.iter)
        micro_statistics.append(optimizer.iter)
    statistics.append(micro_statistics)


print(statistics)
plt.boxplot(statistics, labels=optimizers.keys())
plt.grid()
plt.show()
