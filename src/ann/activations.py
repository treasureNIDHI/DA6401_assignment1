"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

import numpy as np


class Sigmoid:
    
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class Tanh:
    
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, dout):
        return dout * (1 - self.out ** 2)


class ReLU:
    
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    
    def backward(self, dout):
        return dout * self.mask


class Softmax:
    
    def forward(self, x):
        
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out
    
    def backward(self, dout):
        return dout
