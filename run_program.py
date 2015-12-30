execfile('./externalFunctions.py')

from collections import OrderedDict

# Third party library
import numpy as np

# Path to the data files
dataDirectory = "data/"
trainingDataFile = dataDirectory + "train"
wordVectorFile = dataDirectory + "wordVectors_original.txt"
vocabularyFile = dataDirectory + "vocab_original.txt"

# Dimension of each word vector 1*50
wordVectorDimension = 50

# Get index of the words in the vocabulary, count of words in the vocabulary and the word matrix from the vocabulary.
# Each column of the word matrix corresponds to the vector representation of a word in the vocabulary. 
indexedVocabulary, vocabWordCount, wordMatrix = generateWordMatrix(wordVectorFile, vocabularyFile)

all_sentences, all_labels, lastSentence = getSentences(trainingDataFile)
## print lastSentence

windowPadding = 1
contextSize = 3
sentenceTuples = generateWordWindows(windowPadding, indexedVocabulary, all_sentences, all_labels)

# =============
# TEST SCRIPTS
# =============
# Test scripts for the function generateWordWindows()

# Get first sentence
all_sentences[10]

# Get word tuples for the first sentence
for i in range(len(all_sentences[10])):
    print sentenceTuples[i]

# Get index from the indexedVocabulary    
for i in range(len(all_sentences[10])):
    try:
        print all_sentences[10][i].lower(), indexedVocabulary[all_sentences[10][i].lower()]
    except:
        continue
