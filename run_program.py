execfile('./externalFunctions.py')

from collections import OrderedDict

# Third party library
import numpy as np

# Path to the data files
dataDirectory = "data/"
trainingDataFile = dataDirectory + "train"
wordVectorFile = dataDirectory + "wordVectors.txt"
vocabularyFile = dataDirectory +"vocab.txt"

# Dimension of each word vector 1*50
wordVectorDimension = 50

# Get index of the words in the vocabulary, count of words in the vocabulary and the word matrix from the vocabulary.
# Each column of the word matrix corresponds to the vector representation of a word in the vocabulary. 
indexedVocabulary, vocabWordCount, wordMatrix = generateWordMatrix(wordVectorFile, vocabularyFile)


labels, sentences = _split_into_sentences(trainingDataFile)
