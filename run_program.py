execfile('./externalFunctions.py')
execfile('./NeuralNetwork.py')

from collections import OrderedDict
import pickle
# Third party library
#import numpy as np
import NeuralNetwork as nn
#import matplotlib.pyplot as plt



with open('./data_dump/objs.pickle_train_small') as f:
    training_matrix, labelsAsMatrix = pickle.load(f)


ffn = nn.NeuralNetwork(150,200,1)
ffn.SGDbackProp(training_matrix, labelsAsMatrix,20, 0.01)

with open('./output/objs.pickle_nn_300_small', 'w') as f:
    pickle.dump(ffn, f)





