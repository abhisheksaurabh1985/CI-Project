execfile('./externalFunctions.py')

import pickle

from collections import OrderedDict

# Third party library
import numpy as np

# Path to the data files
dataDirectory = "data/"
trainingDataFile = dataDirectory + "train_original" # Change it to train_original before submission
wordVectorFile = dataDirectory + "wordVectors_original.txt"
vocabularyFile = dataDirectory + "vocab_original.txt"

# Dimension of each word vector 1*50
wordVectorDimension = 50

# Get index of the words in the vocabulary, count of words in the vocabulary and the word matrix from the vocabulary.
# Each column of the word matrix corresponds to the vector representation of a word in the vocabulary.
indexedVocabulary, vocabWordCount, vocabWordMatrix = generateWordMatrix(wordVectorFile, vocabularyFile)
print 'Total number of words in the vocabulary: ', vocabWordCount
print '\nProperties of vocabulary word matrix are as under:'
print 'Data type of container: ', type(vocabWordMatrix)
print 'Number of dimensions (aka rank): ', vocabWordMatrix.ndim
print 'Dimensions of the array (row*col): ', vocabWordMatrix.shape
print 'Total number of elements: ', vocabWordMatrix.size

all_sentences, all_labels, lastSentence = getSentences(trainingDataFile)
print '\nNumber of sentences in the training data: ', len(all_sentences)

windowPadding = 1
## contextSize = 3
nestedListSentenceTuples, listSentenceTuples = generateWordWindows(windowPadding, indexedVocabulary, all_sentences, all_labels)

# Generate word vector
training_matrix, labelsAsMatrix = generateWordVectors(wordVectorDimension, vocabWordMatrix, all_labels, windowPadding, listSentenceTuples)
print '\nProperties of training data matrix are as under:'
print 'Data type of container: ', type(training_matrix)
print 'Number of dimensions (aka rank): ', training_matrix.ndim
print 'Dimensions of the array (row*col): ', training_matrix.shape
print 'Total number of elements: ', training_matrix.size

print '\nProperties of training data labels matrix are as under:'
print 'Data type of container: ', type(labelsAsMatrix)
print 'Number of dimensions (aka rank): ', labelsAsMatrix.ndim
print 'Dimensions of the array (row*col): ', labelsAsMatrix.shape
print 'Total number of elements: ', labelsAsMatrix.size


# Store train data into objs.pickle_train file
with open('objs.pickle_train', 'w') as f:
    pickle.dump([training_matrix, labelsAsMatrix], f)