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

def tanhDerivativeVec(y):
    return (1-np.multiply(y,y))

def quadr_error_derivative(o, labels):
    return (np.subtract(o, labels))

def quadr_error_derivative_vec(o,labels):
    return (np.subtract(o,labels))

def store_derivative(o):
    return o*(1-o)

def store_derivative_vec(o):
    return (np.multiply(o,1-o))

def quadr_error(o,y):
    return (1.0/2)*np.power((o-y),2)

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

            self.error_per_epoch = []
            self.costf_per_epoch = []


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
            sum_hidden_neurons = np.dot(self.ai[:,None].T, self.wi)
            # Apply neuron trigger function. ah is the output of the hidden neurons
            # Vectorize is equivalent to map but in numpy
            tanh_fun = np.vectorize(tanh)
            self.ah = tanh_fun(sum_hidden_neurons)
            #self.ah = np.array(map(tanh, sum_hidden_neurons)).T
            self.ah = np.append(self.ah,1)

            # Output activation
            sum_output_neurons = np.dot(self.ah[:,None].T, self.wo)
            # Apply output neuron trigger function. ao is the output of the output layer
            sigmoid_fun = np.vectorize(sigmoid)
            self.ao = sigmoid_fun(sum_output_neurons)
            return self.ao[:]


    def backPropagation(self, training_data, training_labels, number_of_epochs, learning_rate):

        # Matrix of derivatives from the feedforward step for the k hidden units
        D1 = np.zeros(shape=(self.hiddenLayerSize, self.hiddenLayerSize))
        # Matrix of derivatives from the feedforward step for the m output units
        D2 = np.zeros(shape=(self.outputLayerSize, self.outputLayerSize))

        error_per_epoch = 0.0
        for epoch in range(number_of_epochs):
            #learning_rate = learning_rate / (epoch + 1)
            print 'Epoch counter: {}'.format(epoch+1)
            error = 0.0
            cost = []
            error_per_epoch = []

            for data_sample, label_sample in zip(training_data,training_labels):
                # Feedforward computation stemp
                label_sample = np.array(label_sample)
                self.feedForwardNetwork(data_sample)
                # Store the derivatives

                store_derivative_fun = np.vectorize(store_derivative)
                store_tanh_derivative_fun = np.vectorize(tanhDerivative)

                #D22 = np.diag(store_derivative_fun(self.ao))
                D2 = np.diag(store_derivative_vec(self.ao))
                #D2 = np.diag(self.ao)
                #D11 = np.diag(store_tanh_derivative_fun(self.ah[:-1]))
                D1 = np.diag(tanhDerivativeVec(self.ah[:-1]))


                # Derivatives of the quadratic deviations
                qerr_derivative = np.vectorize(quadr_error_derivative)
                #e = np.array(map(quadr_error_derivative, self.ao, np.nditer(label_sample)))
                e = np.array(qerr_derivative(self.ao, label_sample))
                e2 = quadr_error_derivative_vec(self.ao, np.array([float(label_sample)] * self.outputLayerSize))

                #W1 = self.wi[:-1,:]
                #W2 = self.wo[:-1,:]



                # Backpropagated error up to the output units
                delta_output = np.dot(D2, e)
                # Backpropagated error up to the hidden layer
                #delta_hidden = np.dot(D1, W2)#, delta_output)
                delta_hidden = np.dot(D1, self.wo[:-1,:])#, delta_output)
                delta_hidden = np.dot(delta_hidden, delta_output)

                W1_correction = -learning_rate * np.dot(delta_hidden[:,None], self.ai[:,None].T)
                W2_correction = -learning_rate * np.dot(delta_output[:,None], self.ah[:,None].T)

                # Now we adjust the weights
                self.wi = self.wi + W1_correction.T
                self.wo = self.wo + W2_correction.T

                error += quadr_error(self.ao, label_sample)
                cost.append(self.costWithoutRegularization(self.ao, label_sample))
                # backpropagated error up to the output units
                #delta_out = np.dot(D2, e)

                # backpropagated error up to the hidden layer
                #delta_hidden = np.dot(D1)

            #self.error_per_epoch.append(self.insample_error(training_data, training_labels))
            self.costf_per_epoch.append(np.sum(cost))
            #print 'Error: {}'.format(self.error_per_epoch)

        # Feedforward computation
        # Backpropagation to the output layer
        # Backpropagation to the hidden layer
        # Weight updates

       # training_error = error_per_epoch / number_of_epochs
        #print 'training error: {}'.format(error_per_epoch)

        return

    def insample_error(self, training_data, training_labels):

        sample_error = []
        for data_sample, label_sample in zip(training_data, training_labels):
            self.feedForwardNetwork(data_sample)
            sample_error.append(quadr_error(self.ao, label_sample))

        error = (1.0)*(np.sum(sample_error))#/(len(training_labels)))

        return error



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



    def costWithoutRegularization(self, predictedOutput, actualOutput):
        """
        Returns the cost associated with an output prediction.
        np.nan_to_num is used to ensure numerical stability.  If both 'predictedOutput' and 'actualOutput' have a 1.0 in the same
        slot, then the expression (1-actualOutput)*np.log(1-predictedOutput) returns 'nan'.  The np.nan_to_num ensures that 'nan'
        is converted to the correct value (0.0).
        Argument predictedOutput: Output predicted by the network.
        Argument actualOutput: Actual output
        """
        return np.sum(np.nan_to_num(-actualOutput*np.log(predictedOutput)-(1-actualOutput)*np.log(1-predictedOutput)))



    def costWithRegularization(self, paramlambda, predictedOutput, actualOutput):

        J = self.costWithoutRegularization(predictedOutput, actualOutput)
        # np.linalg.norm squares each element in an ndarray row and returns the sum of that row.
        regularizationTerm = (paramlambda/ (2*m)) * [sum(np.linalg.norm(w)**2 for w in self.wi) + sum(np.linalg.norm(u)**2 for u in self.wo)]
        totalCost = J + regularizationTerm
        return totalCost