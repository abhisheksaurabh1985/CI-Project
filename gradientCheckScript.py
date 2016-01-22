import NeuralNetwork as nn
import pickle
import numpy as np

with open('./data_dump/objs.pickle_train_small') as f:
    training_matrix, labelsAsMatrix = pickle.load(f)


check_matrix = training_matrix[0:20,:]
check_labels = labelsAsMatrix[0:20]
net_xor = nn.NeuralNetwork(2,3,1)
train_xor = np.array([[0,0],[0,1],[1,0], [1,1]])
label_xor = np.array([0,1,1,0])
#net_xor.gradientChecking(train_xor, label_xor, 0.1)
net = nn.NeuralNetwork(150,2,1)
net.gradientChecking(check_matrix, check_labels, 0.01)
#net.gradientChecking()