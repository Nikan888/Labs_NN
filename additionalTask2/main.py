import numpy as np
from NeuralNetwork import NeuronLayer, NeuralNetwork

layer1 = NeuronLayer(4, 4)

layer2 = NeuronLayer(2, 4)

layer3 = NeuronLayer(1, 2)

neural_network = NeuralNetwork(layer1, layer2, layer3)

print("1) Random initialized weights: ")
neural_network.print_weights()

training_set_inputs = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 1, 1],
                             [0, 1, 0, 0],
                             [0, 1, 0, 1],
                             [0, 1, 1, 0],
                             [0, 1, 1, 1],
                             [1, 0, 0, 0],
                             [1, 0, 0, 1],
                             [1, 0, 1, 0],
                             [1, 0, 1, 1],
                             [1, 1, 0, 0],
                             [1, 1, 0, 1],
                             [1, 1, 1, 0],
                             [1, 1, 1, 1]])

training_set_outputs = np.array([[1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0]]).T

neural_network.train(training_set_inputs, training_set_outputs, 10000)

print("2) Weights after training: ")
neural_network.print_weights()

print("3) Test: [0, 1, 0, 1] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([0, 0, 0, 0]))
print(output)
print("Test: [0, 0, 0, 1] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([0, 0, 0, 1]))
print(output)
print("Test: [0, 0, 1, 0] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([0, 0, 1, 0]))
print(output)
print("Test: [0, 0, 1, 1] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([0, 0, 1, 1]))
print(output)
print("Test: [0, 1, 0, 0] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([0, 1, 0, 0]))
print(output)
print("Test: [0, 1, 0, 1] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([0, 1, 0, 1]))
print(output)
print("Test: [0, 1, 1, 0] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([0, 1, 1, 0]))
print(output)
print("Test: [0, 1, 1, 1] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([0, 1, 1, 1]))
print(output)
print("Test: [1, 0, 0, 0] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([1, 0, 0, 0]))
print(output)
print("Test: [1, 0, 0, 1] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([1, 0, 0, 1]))
print(output)
print("Test: [1, 0, 1, 0] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([1, 0, 1, 0]))
print(output)
print("Test: [1, 0, 1, 1] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([1, 0, 1, 1]))
print(output)
print("Test: [1, 1, 0, 0] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([1, 1, 0, 0]))
print(output)
print("Test: [1, 1, 0, 1] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([1, 1, 0, 1]))
print(output)
print("Test: [1, 1, 1, 0] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([1, 1, 1, 0]))
print(output)
print("Test: [1, 1, 1, 1] -> ?:")
hidden_state_1, hidden_state_2, output = neural_network.forward(np.array([1, 1, 1, 1]))
print(output)