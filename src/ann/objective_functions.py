"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

class CrossEntropyLoss:

    def forward(self, y_pred, y_true):

        self.y_pred = y_pred
        self.y_true = y_true

        loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

        return loss


    def backward(self):

        gradient = (self.y_pred - self.y_true) / self.y_true.shape[0]


        return gradient


class MSELoss:

    def forward(self, y_pred, y_true):

        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.mean((y_pred - y_true) ** 2)

        return loss


    def backward(self):

        gradient = 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]
        

        return gradient
