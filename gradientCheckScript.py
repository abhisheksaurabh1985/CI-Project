import NeuralNetwork as nn
import pickle
import numpy as np

with open('./data_dump/objs.pickle_train_small') as f:
    training_matrix, labelsAsMatrix = pickle.load(f)


check_matrix = training_matrix[0:20,:]
check_labels = labelsAsMatrix[0:20]
net = nn.NeuralNetwork(150,2,1)
net.gradientChecking(check_matrix, check_labels, 0.01)