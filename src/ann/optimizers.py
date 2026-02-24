"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class SGD:

    def __init__(self, learning_rate):

        self.learning_rate = learning_rate


    def update(self, param, grad):

        updated_param = param - self.learning_rate * grad

        return updated_param

class Momentum:

    def __init__(self, learning_rate, beta):

        self.learning_rate = learning_rate
        self.beta = beta
        self.velocities = {}


    def update(self, param, grad):

        param_id = id(param)

        if param_id not in self.velocities:
            self.velocities[param_id] = np.zeros_like(param)

        self.velocities[param_id] = self.beta * self.velocities[param_id] + (1 - self.beta) * grad
        updated_param = param - self.learning_rate * self.velocities[param_id]

        return updated_param

class RMSProp:
    def __init__(self, learning_rate, beta, epsilon):

        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.squared_gradients = {}

    def update(self, param, grad):

        param_id = id(param)

        if param_id not in self.squared_gradients:
            self.squared_gradients[param_id] = np.zeros_like(param)

        self.squared_gradients[param_id] = self.beta * self.squared_gradients[param_id] + (1 - self.beta) * grad ** 2
        updated_param = param - self.learning_rate * grad / (np.sqrt(self.squared_gradients[param_id]) + self.epsilon)

        return updated_param

class Adam:
    def __init__(self, learning_rate, beta1, beta2, epsilon):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moments = {}
        self.velocities = {}
        self.timesteps = {}
    def update(self, param, grad):

        param_id = id(param)
       

        if param_id not in self.timesteps:
            self.timesteps[param_id] = 0

        self.timesteps[param_id] += 1
        t = self.timesteps[param_id]

        if param_id not in self.moments:
            self.moments[param_id] = np.zeros_like(param)
            self.velocities[param_id] = np.zeros_like(param)

        
        self.moments[param_id] = self.beta1 * self.moments[param_id] + (1 - self.beta1) * grad
        self.velocities[param_id] = self.beta2 * self.velocities[param_id] + (1 - self.beta2) * grad ** 2

        m_hat = self.moments[param_id] / (1 - self.beta1 ** t)
        v_hat = self.velocities[param_id] / (1 - self.beta2 ** t)
        updated_param = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_param

class Nadam:
    def __init__(self, learning_rate, beta1, beta2, epsilon):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moments = {}
        self.velocities = {}
        self.timesteps = {}

    def update(self, param, grad):

        param_id = id(param)
       

        if param_id not in self.timesteps:
            self.timesteps[param_id] = 0

        self.timesteps[param_id] += 1
        t = self.timesteps[param_id]

        if param_id not in self.moments:
            self.moments[param_id] = np.zeros_like(param)
            self.velocities[param_id] = np.zeros_like(param)

        
        self.moments[param_id] = self.beta1 * self.moments[param_id] + (1 - self.beta1) * grad
        self.velocities[param_id] = self.beta2 * self.velocities[param_id] + (1 - self.beta2) * grad ** 2

        m_hat = (self.beta1 * self.moments[param_id] + (1 - self.beta1) * grad) / (1 - self.beta1 ** t)
        v_hat = self.velocities[param_id] / (1 - self.beta2 ** t)
        updated_param = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_param

class NAG:
    def __init__(self, learning_rate, beta):

        self.learning_rate = learning_rate
        self.beta = beta
        self.velocities = {}

    def update(self, param, grad):

        param_id = id(param)

        if param_id not in self.velocities:
            self.velocities[param_id] = np.zeros_like(param)

        lookahead_param = param - self.beta * self.velocities[param_id]
        lookahead_grad = grad 
        self.velocities[param_id] = self.beta * self.velocities[param_id] + (1 - self.beta) * lookahead_grad
        updated_param = param - self.learning_rate * self.velocities[param_id]

        return updated_param