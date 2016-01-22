#### Libraries
## Standard libraries
import math
import random

## Third party libraries
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


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


    def SGDbackProp(self, training_data, training_labels, number_of_epochs, learning_rate):

        for epoch in range(number_of_epochs):
            #learning_rate = learning_rate / (epoch + 1)
            print 'Epoch counter: {}'.format(epoch+1)
            error = 0.0
            cost = []
            error_per_epoch = []

            for data_sample, label_sample in zip(training_data,training_labels):

                W1_correction, W2_correction = self.backPropagation(data_sample, label_sample, learning_rate)
                # Now we adjust the weights
                self.wi = self.wi - learning_rate * W1_correction
                self.wo = self.wo - learning_rate * W2_correction

                error += quadr_error(self.ao, label_sample)
                cost.append(self.costWithoutRegularization(self.ao, label_sample))

                error += quadr_error(self.ao, label_sample)
                cost.append(self.costWithoutRegularization(self.ao, label_sample))

            self.costf_per_epoch.append(np.sum(cost)/len(training_labels))
            print 'cost function per epoch: {}'.format(self.costf_per_epoch[-1])


        return


    def backPropagation(self, data_sample, label_sample, learning_rate):

        #Matrix notation of the backpropagation algorithm from "Neural Networks" Raul Rojas

        # Matrix of derivatives from the feedforward step for the k hidden units
        D1 = np.zeros(shape=(self.hiddenLayerSize, self.hiddenLayerSize))
        # Matrix of derivatives from the feedforward step for the m output units
        D2 = np.zeros(shape=(self.outputLayerSize, self.outputLayerSize))


         # Feedforward computation stemp
        label_sample = np.array(label_sample)
        self.feedForwardNetwork(data_sample)
        # Store the derivatives
        D2 = np.diag(store_derivative_vec(self.ao))
        D1 = np.diag(tanhDerivativeVec(self.ah[:-1]))


        # Derivatives of the quadratic deviations
        qerr_derivative = np.vectorize(quadr_error_derivative)
        e = np.array(qerr_derivative(self.ao, label_sample))

        # Backpropagated error up to the output units
        delta_output = np.dot(D2, e)
        # Backpropagated error up to the hidden layer
        delta_hidden = np.dot(D1, self.wo[:-1,:])#, delta_output)
        delta_hidden = np.dot(delta_hidden, delta_output)

        W1_correction = np.dot(delta_hidden[:,None], self.ai[:,None].T)
        W2_correction = np.dot(delta_output[:,None], self.ah[:,None].T)

        return W1_correction.T, W2_correction.T



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


    def gradientChecking(self, training_data, training_labels, learning_rate, epsilon= 10**-4, errorThreshold = 10**-5):
        '''
        Source: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
        '''
        # Get the gradients from backpropagation
        for training_sample, training_label in zip(training_data, training_labels):
            dW,dU = self.backPropagation(training_sample, training_label, learning_rate)
            gradientsFromBackProp = [dW,dU]
            # List of parameters gradient wrt which is to checked
            modelParameters =  ['W','U']
            for idx, param in enumerate(modelParameters):
                # Get the actual parameter from the model self.wi, self.wo etc.
                if param == 'W':
                    parameter = self.wi#[:-1,:]
                elif param == 'U':
                    parameter = self.wo#[:-1,:]
                print 'Performing gradient check for parameter %s with size %d.' % (param, np.prod(parameter.shape))

                # flags = ['multi_index'] allows indexing into the iterator. Hence, it allows using the index of the current element in computation.
                # op_flags = ['readwrite'] allows modifying the current element. By default, nditer treats the input array as read only object.
                it = np.nditer(parameter, flags = ['multi_index'], op_flags = ['readwrite'])

                while not it.finished:
                    index = it.multi_index
                    # Save the original value of the parameter so that it can be reset later
                    originalParameter = parameter.copy()
                    originalValue = parameter[index]

                    # Gradient for this parameter from backpropagation algorithm
                    gradientFromBPA = gradientsFromBackProp[idx][index]

                    # Gradient calculation using (J(theta + epsilon)- J(theta - epsilon))/(2*h)
                    #parameter[index]= originalValue + epsilon
                    parameter[index]= originalValue + epsilon
                    self.feedForwardNetwork(training_sample)
                    #gradientPlus = self.costWithoutRegularization(self.ao, training_labels[0]) # Doubt in the params which have to be passed.
                    costPlus = quadr_error(self.ao, training_label)
                    #np.copyto(parameter, originalParameter)
                    parameter[index]= originalValue - epsilon
                    self.feedForwardNetwork(training_sample)
                    #gradientMinus = self.costWithoutRegularization(self.ao, training_labels[0]) # Doubt in the params which have to be passed.
                    costMinus = quadr_error(self.ao, training_label)
                    estimatedGradient = (costPlus - costMinus)/ (2*epsilon)

                    # Reset parameter to original value
                    parameter[index]= originalValue

                    # Relative error: (|x - y|/(|x| + |y|))
                    relativeError = np.abs(gradientFromBPA - estimatedGradient) / (np.abs(gradientFromBPA) + np.abs(estimatedGradient))

                    if relativeError >= errorThreshold:
                        print "Gradient Check ERROR: parameter=%s ix=%s" % (param, index)
                        print "+h Loss: %f" % costPlus
                        print "-h Loss: %f" % costMinus
                        print "Estimated_gradient: %f" % estimatedGradient
                        print "Backpropagation gradient: %f" % gradientFromBPA
                        print "Relative Error: %f" % relativeError

                    it.iternext()
                    print "Gradient check for parameter {}_{} passed.".format(param, index)
                    print 'Estimated gradient: {}'.format(estimatedGradient)
                    print 'Gradient from BPA: {}'.format(gradientFromBPA)



    def getPrecisionRecallSupport(yPredicted, yActual):
        '''
        Function returns the precision, recall and F1 score for both the groups i.e. PERSON_NAME and NON_PERSON_NAME.
        INPUT ARGUMENT:
        yPredicted: List containing the predicted label for each word.
        yActual: List containing the actual label for each word.

        OUTPUT ARGUMENT:
        result: Tuple containing the precision, recall, fscore and support for both the negative and the positive class. Each tuple element is a numpy array. Four such arrays in total,
        one each for precision, recall, fscore and support respectively. First element of each of the numpy array is the score for the negative class.
        '''

        # average = None, implies that  scores of both the classes will be returned.
        result = precision_recall_fscore_support(yhat, y, average=None)
        return result
