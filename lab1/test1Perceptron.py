import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, epoches=1000, learning_rate=0.05):
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, training_outputs):
        for i in range(self.epoches):
            for inputs, output in zip(training_inputs, training_outputs):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (output - prediction) * inputs
                self.weights[0] += self.learning_rate * (output - prediction)