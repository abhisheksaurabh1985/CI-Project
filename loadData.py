execfile('./externalFunctions.py')

from collections import OrderedDict

# Third party library
import numpy as np
import NeuralNetwork as nn
#import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pickle

# Path to the data files
dataDirectory = "data/"
trainingDataFile = dataDirectory + "train_original" # Change it to train_original before submission
testDataFile = dataDirectory + "dev"
wordVectorFile = dataDirectory + "wordVectors_original.txt"
vocabularyFile = dataDirectory + "vocab_original.txt"

# Dimension of each word vector 1*50
wordVectorDimension = 50

# Get index of the words in the vocabulary, count of words in the vocabulary and the word matrix from the vocabulary.
# Each column of the word matrix corresponds to the vector representation of a word in the vocabulary.
indexedVocabulary, vocabWordCount, vocabWordMatrix = generateWordMatrix(wordVectorFile, vocabularyFile)
print '================================================================'
print 'Properties of vocabulary and the corresponding matrix'
print '================================================================'
print 'Total number of words in the vocabulary: ', vocabWordCount
print 'Properties of vocabulary word matrix are as under:'
print 'Data type of container: ', type(vocabWordMatrix)
print 'Number of dimensions (aka rank): ', vocabWordMatrix.ndim
print 'Dimensions of the array (row*col): ', vocabWordMatrix.shape
print 'Total number of elements: ', vocabWordMatrix.size

# Define
windowPadding = 1
## contextSize = 3

'''
Load training data
'''
train_all_sentences, train_all_labels, train_lastSentence = getSentences(trainingDataFile)
# Get length of each sentence in training data and also the labels for each word in a sentence.
trainingSentencesLength, trainingSentencesLabels = getLabelsForeachWordInSentence(train_all_sentences, train_all_labels)
print '\nNumber of sentences in the training data: ', len(train_all_sentences)

##nestedListSentenceTuples, listSentenceTuples = generateWordWindows(windowPadding, indexedVocabulary, train_all_sentences, train_all_labels)
train_nestedListWordWindowsByIndex, train_listWordWindowsByIndex = generateWordWindows(windowPadding, indexedVocabulary, train_all_sentences, train_all_labels)

# Generate word vector
training_matrix, training_labels_matrix = generateWordVectors(wordVectorDimension, vocabWordMatrix, train_all_labels, windowPadding, train_listWordWindowsByIndex)

# Get properties of the training data set
print '================================================================'
print 'Properties of training data matrix are as under:'
print '================================================================'
print 'Data type of container: ', type(training_matrix)
print 'Number of dimensions (aka rank): ', training_matrix.ndim
print 'Dimensions of the array (row*col): ', training_matrix.shape
print 'Total number of elements: ', training_matrix.size
print '================================================================'
print 'Properties of training data labels matrix are as under:'
print '================================================================'
print 'Data type of container: ', type(training_labels_matrix)
print 'Number of dimensions (aka rank): ', training_labels_matrix.ndim
print 'Dimensions of the array (row*col): ', training_labels_matrix.shape
print 'Total number of elements: ', training_labels_matrix.size


'''
Load test data
'''
test_all_sentences, test_all_labels, test_lastSentence = getSentences(testDataFile)
# Get length of each sentence in test data and also the labels for each word in a sentence.
testSentencesLength, testSentencesLabels = getLabelsForeachWordInSentence(test_all_sentences, test_all_labels)

print '\nNumber of sentences in the test data: ', len(test_all_sentences)

##nestedListSentenceTuples, listSentenceTuples = generateWordWindows(windowPadding, indexedVocabulary, test_all_sentences, test_all_labels)
test_nestedListWordWindowsByIndex, test_listWordWindowsByIndex = generateWordWindows(windowPadding, indexedVocabulary, test_all_sentences, test_all_labels)

# Generate word vector
test_matrix, test_labels_matrix = generateWordVectors(wordVectorDimension, vocabWordMatrix, test_all_labels, windowPadding, test_listWordWindowsByIndex)

# Get properties of the test data set
print '================================================================'
print 'Properties of test data matrix are as under:'
print '================================================================'
print 'Data type of container: ', type(test_matrix)
print 'Number of dimensions (aka rank): ', test_matrix.ndim
print 'Dimensions of the array (row*col): ', test_matrix.shape
print 'Total number of elements: ', test_matrix.size
print '================================================================'
print 'Properties of test data labels matrix are as under:'
print '================================================================'
print 'Data type of container: ', type(test_labels_matrix)
print 'Number of dimensions (aka rank): ', test_labels_matrix.ndim
print 'Dimensions of the array (row*col): ', test_labels_matrix.shape
print 'Total number of elements: ', test_labels_matrix.size


'''
Toy dataset to check implementation and execution time. This dataset has the same number of features as the training and test data. 100 samples across two classes have been generated.
'''
X1, Y1 = make_classification(n_samples= 100, n_features=150, n_redundant=0, n_informative=150, n_repeated= 0, n_clusters_per_class=1)



# Store train data into objs.pickle_train file
with open('./data_dump/objs.pickle_train', 'w') as f:
    pickle.dump([training_matrix, training_labels_matrix], f)


with open('./data_dump/objs.pickle_test', 'w') as f:
    pickle.dump([test_matrix, test_labels_matrix], f)