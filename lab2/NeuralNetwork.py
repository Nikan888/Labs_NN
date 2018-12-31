import numpy as np

class NeuronLayer(object):
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork(object):
    def __init__(self, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output_from_layer_1, output_from_layer_2, output_from_layer_3, output_from_layer_4, output_from_layer_5, output_from_layer_6, output_from_layer_7, output_from_layer_8 = self.forward(training_set_inputs)

            layer8_error = training_set_outputs - output_from_layer_8
            layer8_delta = layer8_error * self.__sigmoid_derivative(output_from_layer_8)

            layer7_error = layer8_delta.dot(self.layer8.weights.T)
            layer7_delta = layer7_error * self.__sigmoid_derivative(output_from_layer_7)

            layer6_error = layer7_delta.dot(self.layer7.weights.T)
            layer6_delta = layer6_error * self.__sigmoid_derivative(output_from_layer_6)

            layer5_error = layer6_delta.dot(self.layer6.weights.T)
            layer5_delta = layer5_error * self.__sigmoid_derivative(output_from_layer_5)

            layer4_error = layer5_delta.dot(self.layer5.weights.T)
            layer4_delta = layer4_error * self.__sigmoid_derivative(output_from_layer_4)

            layer3_error = layer4_delta.dot(self.layer4.weights.T)
            layer3_delta = layer3_error * self.__sigmoid_derivative(output_from_layer_3)

            layer2_error = layer3_delta.dot(self.layer3.weights.T)
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            layer1_error = layer2_delta.dot(self.layer2.weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)
            layer4_adjustment = output_from_layer_3.T.dot(layer4_delta)
            layer5_adjustment = output_from_layer_4.T.dot(layer5_delta)
            layer6_adjustment = output_from_layer_5.T.dot(layer6_delta)
            layer7_adjustment = output_from_layer_6.T.dot(layer7_delta)
            layer8_adjustment = output_from_layer_7.T.dot(layer8_delta)

            self.layer1.weights += layer1_adjustment
            self.layer2.weights += layer2_adjustment
            self.layer3.weights += layer3_adjustment
            self.layer4.weights += layer4_adjustment
            self.layer5.weights += layer5_adjustment
            self.layer6.weights += layer6_adjustment
            self.layer7.weights += layer7_adjustment
            self.layer8.weights += layer8_adjustment

    def forward(self, inputs):
        output_from_layer1 = self.__sigmoid(np.dot(inputs, self.layer1.weights))
        output_from_layer2 = self.__sigmoid(np.dot(output_from_layer1, self.layer2.weights))
        output_from_layer3 = self.__sigmoid(np.dot(output_from_layer2, self.layer3.weights))
        output_from_layer4 = self.__sigmoid(np.dot(output_from_layer3, self.layer4.weights))
        output_from_layer5 = self.__sigmoid(np.dot(output_from_layer4, self.layer5.weights))
        output_from_layer6 = self.__sigmoid(np.dot(output_from_layer5, self.layer6.weights))
        output_from_layer7 = self.__sigmoid(np.dot(output_from_layer6, self.layer7.weights))
        output_from_layer8 = self.__sigmoid(np.dot(output_from_layer7, self.layer8.weights))
        return output_from_layer1, output_from_layer2, output_from_layer3, output_from_layer4, output_from_layer5, \
               output_from_layer6, output_from_layer7, output_from_layer8

    def print_weights(self):
        print("    Layer 1 (144 neurons, each with 25 inputs): ")
        print(self.layer1.weights)
        print("    Layer 2 (72 neurons, with 144 inputs):")
        print(self.layer2.weights)
        print("    Layer 3 (36 neurons, with 72 inputs):")
        print(self.layer3.weights)
        print("    Layer 4 (18 neurons, with 36 inputs):")
        print(self.layer4.weights)
        print("    Layer 5 (9 neurons, with 18 inputs):")
        print(self.layer5.weights)
        print("    Layer 6 (4 neurons, with 9 inputs):")
        print(self.layer6.weights)
        print("    Layer 7 (2 neurons, with 4 inputs):")
        print(self.layer7.weights)
        print("    Layer 8 (1 neuron, with 2 inputs):")
        print(self.layer8.weights)