execfile('./externalFunctions.py')
execfile('./NeuralNetwork.py')

from collections import OrderedDict
import pickle
# Third party library
#import numpy as np
import NeuralNetwork as nn
#import matplotlib.pyplot as plt



with open('./data_dump/objs.pickle_train') as f:
    training_matrix, labelsAsMatrix = pickle.load(f)


ffn = nn.NeuralNetwork(150,200,1)
ffn.SGDbackProp(training_matrix, labelsAsMatrix,5, 0.01)



