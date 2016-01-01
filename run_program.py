execfile('./externalFunctions.py')

from collections import OrderedDict

# Third party library
import numpy as np

# Path to the data files
dataDirectory = "data/"
trainingDataFile = dataDirectory + "train" # Change it to train_original before submission
wordVectorFile = dataDirectory + "wordVectors_original.txt"
vocabularyFile = dataDirectory + "vocab_original.txt"

# Dimension of each word vector 1*50
wordVectorDimension = 50

# Get index of the words in the vocabulary, count of words in the vocabulary and the word matrix from the vocabulary.
# Each column of the word matrix corresponds to the vector representation of a word in the vocabulary. 
indexedVocabulary, vocabWordCount, vocabWordMatrix = generateWordMatrix(wordVectorFile, vocabularyFile)

all_sentences, all_labels, lastSentence = getSentences(trainingDataFile)
print lastSentence

windowPadding = 1
## contextSize = 3
nestedListSentenceTuples, listSentenceTuples = generateWordWindows(windowPadding, indexedVocabulary, all_sentences, all_labels)

# Generate word vector
training_matrix, labelsAsMatrix = generateWordVectors(wordVectorDimension, vocabWordMatrix, all_labels, windowPadding, listSentenceTuples)
