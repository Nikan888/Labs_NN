from Perceptron import Perceptron
from ImageProcessor import *

#training_inputs_logAnd = np.array([ [0, 0],
#                                    [0, 1],
#                                    [1, 0],
#                                    [1, 1],
#                                 ])

#training_outputs_logAnd = np.array([0, 0, 0, 1])

#perceptron_logAnd = Perceptron(2)

#perceptron_logAnd.train(training_inputs_logAnd, training_outputs_logAnd)

#print(perceptron_logAnd.weights)

#testInput1 = np.array([1, 1])
#print(perceptron_logAnd.predict(testInput1))

#testInput2 = np.array([0, 1])
#print(perceptron_logAnd.predict(testInput2))

#training_inputs_smiles = np.array([
#    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
#    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
#    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
#    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
#    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
#])

#training_outputs_smiles = np.array([1, 0, 1, 1, 0])

training_inputs_smiles, test_inputs_smiles = getImages()

#training_outputs_smiles = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

training_outputs_smiles = getAnswers()

perceptron_smiles = Perceptron(25)

perceptron_smiles.train(training_inputs_smiles, training_outputs_smiles)

#print(perceptron_smiles.weights)

#testInput1 = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1])
#print(perceptron_smiles.predict(testInput1))

#testInput2 = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0])
#print(perceptron_smiles.predict(testInput2))

for test_input in test_inputs_smiles:
    print(perceptron_smiles.predict(test_input))