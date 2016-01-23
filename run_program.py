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


ffn = nn.NeuralNetwork(150,100,1)
ffn.SGDbackProp(training_matrix, labelsAsMatrix,5, 0.1, 5)

#with open('./output/objs.pickle_nn_200_3e_small', 'w') as f:
#    pickle.dump(ffn, f)





