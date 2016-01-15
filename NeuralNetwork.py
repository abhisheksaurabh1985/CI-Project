#### Libraries
## Standard libraries
import math
import random

## Third party libraries
import numpy as np

# Miscellaneous functions
## Sigmoid function for the output layer
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

## Derivative of sigmoid to be used in Backpropagation algorithm.
def sigmoidDerivative(y):
    return y* (1- y)

## Hyperbolic tanh function for the hidden layer
def tanh(x):
    return math.tanh(x)

def tanhDerivative(y):
    return (1- y*y)

class NeuralNetwork(object):
    """
    Consists of three layers: input, hidden and output. The size of hidden layer is user defined when initializing the network.
    """
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize):
            """
            Argument input: Number of input neurons
            Argument hidden: Number of hidden neurons
            Argument output: Number of output neurons
            """
            # Initialize arrays for inputs
            self.inputLayerSize = inputLayerSize + 1 # add 1 for bias
            self.hiddenLayerSize = hiddenLayerSize
            self.outputLayerSize = outputLayerSize

            # Set up array of 1s for activation
            self.ai = np.array([1.0] * self.inputLayerSize)
            self.ah = np.array([1.0] * self.hiddenLayerSize)
            self.ao = np.array([1.0] * self.outputLayerSize)


            # Random weights
            self.wi, self.wo = self.randomlyInitializeParameters()


    def feedForwardNetwork(self, inputs):
            """
            Loops through each node of the hidden layer. Adds the output from input layer times weight. Hyperbolic tanh is used as an activation function at the hidden layer.
            Likewise, loops through each node of the output layer. Adds the output from the hidden layer times the weight of the edges from hidden node to the output node.
            Sigmoid function is applied on this summation at each output node.
            Argument inputs: input data
            Output argument: Updated activation output vector
            """
            if len(inputs) != self.inputLayerSize - 1: # inputLayer in the __init__ includes a term for bias. Hence, -1 has to be subtracted.
                raise ValueError('Wrong number of inputs.')

            # input activations
            for i in range(self.inputLayerSize - 1): # -1 is to avoid the bias. Nope.
                self.ai[i] = inputs[i]
            # Hidden activation
            sum_hidden_neurons = np.dot(self.ai.T, self.wi)
            # Apply neuron trigger function. ah is the output of the hidden neurons
            self.ah = np.array(map(tanh, sum_hidden_neurons)).T
            self.ah = np.append(self.ah,1)

            # Output activation
            sum_output_neurons = np.dot(self.ah.T, self.wo)
            # Apply output neuron trigger function. ao is the output of the output layer
            self.ao = map(sigmoid, sum_output_neurons)
            return self.ao[:]


    def backPropagation(self, training_data, training_labels, number_of_epochs):

        # Matrix of derivatives from the feedforward step for the k hidden units
        D1 = np.zeros(shape=(self.hiddenLayerSize, self.hiddenLayerSize))
        # Matrix of derivatives from the feedforward step for the m output units
        D2 = np.zeros(shape=(self.outputLayerSize, self.outputLayerSize))

        for epoch in range(number_of_epochs):

            for data_sample in training_data:
                # Feedforward computation stemp
                self.feedForwardNetwork(data_sample)
                # Store the derivatives
                D1 = np.diag(map(tanhDerivative, self.ah[:-1]))
                D2 = np.diag(map(sigmoidDerivative, self.ao))
                #D1 = np.diag((self.ah).subtract())
                # Derivatives of the quadratic deviations
                #e = np.subtract(ao, training_labels)

                W1 = self.wi[:-1,:]
                W2 = self.wo[:-1,:]

                o1 = self.ai


                # backpropagated error up to the output units
                #delta_out = np.dot(D2, e)

                # backpropagated error up to the hidden layer
                #delta_hidden = np.dot(D1)

        # Feedforward computation
        # Backpropagation to the output layer
        # Backpropagation to the hidden layer
        # Weight updates

        return

    def randomlyInitializeParameters(self):
            epsilon = round(math.sqrt(6)/ math.sqrt(self.inputLayerSize + self.hiddenLayerSize),2)

            # Generate a random array of floats between 0 and 1
            #weightsLayerOne= np.random.random((self.hiddenLayerSize, self.inputLayerSize)) # 1 for bias in self.input has already been accounted in __init__ method.
            weightsLayerOne= np.random.random((self.inputLayerSize, self.hiddenLayerSize)) # 1 for bias in self.input has already been accounted in __init__ method.
            # Normalize so that it spans a range of twice epsilon
            weightsLayerOne = weightsLayerOne * 2* epsilon

            # Shift so that mean is at zero
            weightsLayerOne = weightsLayerOne - epsilon

            # In the similar way randomly generate the weights for the connection between hidden and the output layer
            #weightsLayerTwo = np.random.random((self.outputLayerSize, self.hiddenLayerSize + 1)) * 2 * epsilon - epsilon
            weightsLayerTwo = np.random.random((self.hiddenLayerSize + 1, self.outputLayerSize)) * 2 * epsilon - epsilon

            return weightsLayerOne, weightsLayerTwo

    def SGD(self, trainingData, numberOfEpochs, miniBatchSize, learningRataEta, regularizationParamlambda= 0.0):

            lengthTrainingData = len(trainingData)

            for i in xrange(numberOfEpochs):
                random.shuffle(trainingData)
                miniBatches = [trainingData[k: k + miniBatchSize] for k in xrange(0, n, miniBatchSize)]
                for miniBatch in miniBatches:
                    self.updateMiniBatch(miniBatch, learningRateEta, regularizationParamlambda, len(trainingData))
                print 'Epoch %s training complete' % j
