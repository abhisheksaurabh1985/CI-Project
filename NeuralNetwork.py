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
def sigmoidDerivative(o):
    return (np.multiply(o,1-o))

## Hyperbolic tanh function for the hidden layer
def tanh(x):
    return math.tanh(x)

def tanhDerivative(y):
    return (1- y*y)

def tanhDerivativeVec(y):
    return (1-np.multiply(y,y))

def quadr_error_derivative(o, labels):
    return (np.subtract(o, labels))


def cross_entropy_derivative(o, labels):
    return (np.divide(np.subtract(o,labels), (np.multiply(o, (1-o)))))

def store_derivative(o):
    return o*(1-o)

def store_derivative_vec(o):
    return (np.multiply(o,1-o))

def delta(o, y):
    return(o-y)


def quadr_error(o,y):
    return (1.0/2)*np.power((o-y),2)


class quadraticErrorJ(object):

    @staticmethod
    def evaluate(actualOutput,predictedOutput):

        return (1.0/2)*np.power((predictedOutput-actualOutput),2)

    @staticmethod
    def evaluate_derivative(actualOutput, predictedOutput):
        '''

        :param o: output of the neural network
        :param labels: labels of the given data
        :return: evaluation of the derivative
        '''
        return (np.divide(np.subtract(predictedOutput,actualOutput), (np.multiply(predictedOutput, (1-predictedOutput)))))
        return (np.subtract(o, labels))



class cxEntropyJ(object):

    @staticmethod
    def evaluate(actualOutput,predictedOutput, wi, wo, lambda_reg):
        """
        Returns the cost associated with an output prediction.
        np.nan_to_num is used to ensure numerical stability.  If both 'predictedOutput' and 'actualOutput' have a 1.0 in the same
        slot, then the expression (1-actualOutput)*np.log(1-predictedOutput) returns 'nan'.  The np.nan_to_num ensures that 'nan'
        is converted to the correct value (0.0).
        Argument predictedOutput: Output predicted by the network.
        Argument actualOutput: Actual output
        """
        cost = np.sum(np.nan_to_num(-actualOutput*np.log(predictedOutput)-(1-actualOutput)*np.log(1-predictedOutput)))
        reg_term = ((1.0)*lambda_reg/2) * ((np.square(wi[:-1,:])).sum() + np.square(wo[:-1,:]).sum())
        cost += reg_term

        return cost

    @staticmethod
    def evaluate_derivative(actualOutput, predictedOutput):#, wi, wo, lambda_reg):
        '''

        :param o: output of the neural network
        :param labels: labels of the given data
        :return: evaluation of the derivative
        '''

        cx_entropy_derivative = np.divide(np.subtract(predictedOutput,actualOutput), (np.multiply(predictedOutput,
                                            (1-predictedOutput))))

        return cx_entropy_derivative

    @staticmethod
    def reg_derivative(wi,wo,lambda_reg):

        dw1 = np.zeros(wi.shape)
        dw2 = np.zeros(wo.shape)
        dw1[:-1,:] = ((1.0)*lambda_reg)*(wi[:-1,:])
        dw2[:-1,:] = ((1.0)*lambda_reg)*(wo[:-1,:])
        return dw1,dw2

    # @staticmethod
    # def regularization_term(wi,wo,lambda_reg):
    #     reg_term_wi = np.zeros(wi.shape)
    #     reg_term_wo = np.zeros(wo.shape)
    #
    #     reg_term_wi[:-1,:] = wi[:-1,:] * lambda_reg/(reg_term_wi[:-1,:].size)
    #     reg_term_wo[:-1,:] = wo[:-1,:] * lambda_reg/(reg_term_wo[:-1,:].size)
    #     return reg_term_wi, reg_term_wo





class NeuralNetwork(object):
    """
    Consists of three layers: input, hidden and output. The size of hidden layer is user defined when initializing the network.
    """
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize, costFunction=cxEntropyJ):
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

            # Cost function for the network
            self.costFunction = costFunction


            # Random weights
            self.wi, self.wo = self.randomlyInitializeParameters()

            # Train values
            self.costf_per_epoch = []
            self.accuracy_per_epoch = []

            # Validation values
            self.val_costf_per_epoch = []
            self.val_accuracy = []


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
            # Added input for bias term in hidden units
            self.ah = np.append(self.ah,1)

            # Output activation
            sum_output_neurons = np.dot(self.ah[:,None].T, self.wo)
            # Apply output neuron trigger function. ao is the output of the output layer
            sigmoid_fun = np.vectorize(sigmoid)
            self.ao = sigmoid_fun(sum_output_neurons)
            return self.ao[:]


    def SGDbackProp(self, training_data, training_labels, number_of_epochs,
                    learning_rate,
                    lambda_reg=0,
                    monitor=False,
                    validation_data=None,
                    validation_labels=None):

        '''
        Stochastic Gradient Descent Backpropagation Algorithm.


        :param training_data:
        :param training_labels:
        :param number_of_epochs:
        :param learning_rate:
        :return:
        '''


        for epoch in range(number_of_epochs):
            #learning_rate = learning_rate / (epoch + 1)
            print 'Epoch counter: {}'.format(epoch+1)
            cost = []


            for data_sample, label_sample in zip(training_data,training_labels):

                dW1, dW2 = self.backPropagation(data_sample, label_sample, learning_rate, lambda_reg)
                # Now we adjust the weights
                self.wi = self.wi - learning_rate * dW1
                self.wo = self.wo - learning_rate * dW2

                cost.append(self.costFunction.evaluate(label_sample, self.ao,self.wi,self.wo,lambda_reg))


            self.costf_per_epoch.append(np.sum(cost)/len(training_labels))
            print 'cost function per epoch: {}'.format(self.costf_per_epoch[-1])

            #if monitor:
            #    predicted_labels_train = self.testSample(training_data)
            #    self.accuracy_per_epoch.append(self.getPrecisionRecallSupport(predicted_labels_train,training_labels))
                #predicted_labels_validation = self.testSample(validation_data)
                #self.val_accuracy.append(self.getPrecisionRecallSupport(predicted_labels_validation, validation_labels))

        return


    def backPropagation(self, data_sample, label_sample, learning_rate, lambda_reg=0):

        #Matrix notation of the backpropagation algorithm from "Neural Networks" Raul Rojas


         # Feedforward computation step
        label_sample = np.array(label_sample)
        self.feedForwardNetwork(data_sample)
        # Store the derivatives of output units
        D2 = np.diag(sigmoidDerivative(self.ao))
        # Store the derivatives of hidden units
        D1 = np.diag(tanhDerivativeVec(self.ah[:-1]))


        # Derivatives of the cost function with respect to the output units
        derivative_J_output = np.vectorize(self.costFunction.evaluate_derivative)
        e = np.array(derivative_J_output(label_sample, self.ao))

        # Backpropagated error up to the output units
        delta_output = np.dot(D2, e)
        # Backpropagated error up to the hidden layer
        delta_hidden = np.dot(D1, self.wo[:-1,:])
        delta_hidden = np.dot(delta_hidden, delta_output)

        dw1, dw2 = self.costFunction.reg_derivative(self.wi, self.wo, lambda_reg)

        # Gradient of the cost function with respect to the weights of output and hidden units
        dW1 = np.dot(delta_hidden[:,None], self.ai[:,None].T).T + dw1
        dW2 = np.dot(delta_output[:,None], self.ah[:,None].T).T + dw2

        return dW1, dW2



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


    def gradientChecking(self, training_data, training_labels, learning_rate, lambda_reg=0, epsilon= 10**-4, errorThreshold = 10**-5):
        '''
        Source: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
        '''
        # Get the gradients from backpropagation
        for training_sample, training_label in zip(training_data, training_labels):
            dW,dU = self.backPropagation(training_sample, training_label, learning_rate, lambda_reg)
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

                it = np.nditer(parameter, flags = ['multi_index'], op_flags = ['readwrite'])

                while not it.finished:
                    index = it.multi_index
                    # Save the original value of the parameter so that it can be reset later
                    originalValue = parameter[index]

                    # Gradient for this parameter from backpropagation algorithm
                    gradientFromBPA = gradientsFromBackProp[idx][index]

                    # Gradient calculation using (J(theta + epsilon)- J(theta - epsilon))/(2*h)
                    # Modify parameter with +epsilon and compute the cost function
                    parameter[index]= originalValue + epsilon
                    self.feedForwardNetwork(training_sample)
                    costPlus = self.costFunction.evaluate(training_label, self.ao,self.wi,self.wo,lambda_reg)

                    # Modify parameter with -epsilon and compute cost function
                    parameter[index]= originalValue - epsilon
                    self.feedForwardNetwork(training_sample)
                    costMinus = self.costFunction.evaluate(training_label, self.ao,self.wi, self.wo, lambda_reg)


                    estimatedGradient = (costPlus - costMinus)/ (2*epsilon)

                    # Reset parameter to original value
                    parameter[index]= originalValue

                    # Relative error: (|x - y|/(|x| + |y|))
                    relativeError = np.abs(gradientFromBPA - estimatedGradient) / (np.abs(gradientFromBPA) + np.abs(estimatedGradient))

                    if relativeError >= errorThreshold:
                        print "**************************"
                        print "Gradient Check ERROR: parameter=%s ix=%s" % (param, index)
                        print "+h Loss: %f" % costPlus
                        print "-h Loss: %f" % costMinus
                        print "Estimated_gradient: %f" % estimatedGradient
                        print "Backpropagation gradient: %f" % gradientFromBPA
                        print "Relative Error: %f" % relativeError
                        print "**************************"

                    else:
                        print "Gradient check for parameter {}_{} passed.".format(param, index)
                        print 'Estimated gradient: {}'.format(estimatedGradient)
                        print 'Gradient from BPA: {}'.format(gradientFromBPA)

                    it.iternext()




    def getPrecisionRecallSupport(self,yPredicted, yActual):
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
        result = precision_recall_fscore_support(yPredicted, yActual, average=None)
        return result



    def testSample(self, sample_data):

        predicted_labels = []

        for sample in sample_data:
            self.feedForwardNetwork(sample)
            predicted_labels.append(1 if self.ao>0.5 else 0)

        return predicted_labels