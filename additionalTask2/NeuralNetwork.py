import numpy as np

np.random.seed(29)


class NeuronLayer(object):
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork(object):
    def __init__(self, layer1, layer2, layer3):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output_from_layer_1, output_from_layer_2, output_from_layer_3 = self.forward(training_set_inputs)

            layer3_error = training_set_outputs - output_from_layer_3
            layer3_delta = layer3_error * self.__sigmoid_derivative(output_from_layer_3)

            layer2_error = layer3_delta.dot(self.layer3.weights.T)
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            layer1_error = layer2_delta.dot(self.layer2.weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)

            self.layer1.weights += layer1_adjustment
            self.layer2.weights += layer2_adjustment
            self.layer3.weights += layer3_adjustment

    def forward(self, inputs):
        output_from_layer1 = self.__sigmoid(np.dot(inputs, self.layer1.weights))
        output_from_layer2 = self.__sigmoid(np.dot(output_from_layer1, self.layer2.weights))
        output_from_layer3 = self.__sigmoid(np.dot(output_from_layer2, self.layer3.weights))
        return output_from_layer1, output_from_layer2, output_from_layer3

    def print_weights(self):
        print("    Layer 1 (4 neurons, each with 4 inputs): ")
        print(self.layer1.weights)
        print("    Layer 2 (2 neurons, with 4 inputs):")
        print(self.layer2.weights)
        print("    Layer 3 (1 neuron, with 2 inputs):")
        print(self.layer3.weights)