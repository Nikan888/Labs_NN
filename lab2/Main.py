from NeuralNetwork import  *
from ImageProcessor import *

layer1 = NeuronLayer(144, 25)

layer2 = NeuronLayer(72, 144)

layer3 = NeuronLayer(36, 72)

layer4 = NeuronLayer(18, 36)

layer5 = NeuronLayer(9, 18)

layer6 = NeuronLayer(4, 9)

layer7 = NeuronLayer(2, 4)

layer8 = NeuronLayer(1, 2)

neural_network = NeuralNetwork(layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8)

print("1) Random initialized weights: ")
neural_network.print_weights()

#training_set_inputs = np.array([[0, 0, 0, 0],
#                             [0, 0, 0, 1],
#                             [0, 0, 1, 0],
#                             [0, 0, 1, 1],
#                             [0, 1, 0, 0],
#                             [0, 1, 0, 1],
#                             [0, 1, 1, 0],
#                             [0, 1, 1, 1],
#                             [1, 0, 0, 0],
#                             [1, 0, 0, 1],
#                             [1, 0, 1, 0],
#                             [1, 0, 1, 1],
#                             [1, 1, 0, 0],
#                             [1, 1, 0, 1],
#                             [1, 1, 1, 0],
#                             [1, 1, 1, 1]])

#training_set_outputs = np.array([[1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0]]).T

#training_inputs_images = np.array([
#    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
#    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
#    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
#    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
#    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
#])

#training_outputs_images = np.array([[1, 0, 1, 1, 0]]).T

training_inputs_images, test_inputs_images = getImages()

training_outputs_images = np.array([[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]]).T
#training_outputs_images = getAnswers()

neural_network.train(training_inputs_images, training_outputs_images, 60000)

print("2) Weights after training: ")
neural_network.print_weights()

for test_input in test_inputs_images:
    hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, output = neural_network.forward(test_input)
    print(output)

#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, output = neural_network.forward(np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, output = neural_network.forward(np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]]))
#print(output)

#print("//Stage 3) Test: [0, 1, 0, 1] -> ?: //")
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([0, 0, 0, 0]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([0, 0, 0, 1]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([0, 0, 1, 0]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([0, 0, 1, 1]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([0, 1, 0, 0]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([0, 1, 0, 1]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([0, 1, 1, 0]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([0, 1, 1, 1]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([1, 0, 0, 0]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([1, 0, 0, 1]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([1, 0, 1, 0]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([1, 0, 1, 1]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([1, 1, 0, 0]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([1, 1, 0, 1]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([1, 1, 1, 0]))
#print(output)
#hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6, hidden_state_7, hidden_state_8, output = neural_network.forward(np.array([1, 1, 1, 1]))
#print(output)