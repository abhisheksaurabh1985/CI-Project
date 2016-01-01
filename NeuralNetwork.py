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
	def __init__(self, input, hidden, output):
		"""
		Argument input: Number of input neurons
		Argument hidden: Number of hidden neurons
		Argument output: Number of output neurons
		"""
		# Initialize arrays for inputs
		self.input = input + 1 # add 1 for bias 
		self.hidden = hidden
		self. output = output

		# Set up array of 1s for activation
		self.ai = [1.0] * self.input
		self.ah = [1.0] * self.hidden
		self.ao = [1.0] * self.output

        def feedForwardNetwork(self, inputs):
                """
                Loops through each node of the hidden layer. Adds the output from previous layer times weight. Sigmoid function is applied on this summation
                at each node. Output from a node hence obtained is passed on to the next layer.
                Argument inputs: input data
                Output argument: updated activation output vector
                """                
	def randomlyInitializeParameters(self):
		epsilon = round(math.sqrt(6)/ math.sqrt(self.input + self.hidden),2)
		# Generate a random array of floats between 0 and 1
		weightsLayerOne= numpy.random.random((self.hidden, self.input)) # 1 for bias in self.input has already been accounted in __init__ method.
		# Normalize so that it spans a range of twice epsilon
		weightsLayerOne = weightsLayerOne * 2* epsilon
		# Shift so that mean is at zero
		weightsLayerOne = weightsLayerOne - epsilon
		# In the similar way randomly generate the weights for the connection between hidden and the output layer
		weightsLayerTwo = numpy.random.random((self.output, self.hidden + 1)) * 2 * epsilon - epsilon
		return weightsLayerOne, weightsLayerTwo		


        




