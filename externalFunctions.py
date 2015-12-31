def generateWordMatrix(fileWordVector, fileVocab):
    indexVocabulary = OrderedDict()
    ##def generateWordMatrix(wordVectorFile, vocabularyFile):
    with open(wordVectorFile, 'r') as fileObj1, open(vocabularyFile, 'r') as fileObj2:
        for eachIndexInVocab, eachWordInVocab in enumerate(fileObj2):
            indexVocabulary[eachWordInVocab.strip('\n')]= eachIndexInVocab
        
        # Number of words in vocabulary
        numWords = eachIndexInVocab + 1

        # Define word matrix
        wordMatrix = np.zeros(shape= (wordVectorDimension, numWords))

        for eachIndexInWordVector, eachWordVector in enumerate(fileObj1):
            features = np.array(eachWordVector.split())
            # print eachIndexInWordVector
            wordMatrix[:,eachIndexInWordVector]= features.T
        
        return indexVocabulary, numWords, wordMatrix
        

# Function creates sentences from the training set. Removes the labels associated with each word and appends the words to form a sentence, until word is an end-of-the-line
# character.
def getSentences(trainingDataFileName):

    # Define end-of line characters
    endOfLineCharacters = '.?!'
    # List to store sentences and labels corresponding to each word.
    allSentences = []
    allLabels = []

    with open(trainingDataFileName, 'r') as fileObj:
        sentence = []
        for line in fileObj:
            if len(line.split())== 2:
                try:
                    word, label = line.split()
                except:
                    continue
                sentence = sentence + [word]    

                # Training data has labels 'PERSON' and 'O' for name and non-name word respectively. Codify it as 1 and 0.
                if label == 'PERSON':
                    newLabel = 1
                else:
                    newLabel = 0
                allLabels = allLabels + [newLabel]

                if word in endOfLineCharacters:
                    allSentences = allSentences + [sentence]
                    sentence = []

        # Last sentence does not end with any of the end of line characters defined earlier. Add it to the list of sentences.
        if sentence!= []:
            allSentences = allSentences + [sentence]

        return allSentences, allLabels, sentence    

# Function returns a window of words formed from a sentence which shall be fed into the NN. To include the context of the first and the last word
# of a sentence special start and end token, <s> and </s> is appended to every sentence. For each word window, the function returns the index from the
# indexed dictionary created from the vocabulary by the function generateWordMatrix.
def generateWordWindows(windowPadding, dictionaryIndex, listAllSentences, listAllLabels):
    indexStartToken = dictionaryIndex['<s>']
##    print indexStartToken
    indexEndToken = dictionaryIndex['</s>']
##    print indexEndToken
    listListWindows = []
    for eachSentence in listAllSentences:
        lengthEachSentence = len(eachSentence)
        listEachSentenceWindow = []
        for i in range(lengthEachSentence):
            wordWithContext = []
            for j in range(i-windowPadding, i + windowPadding + 1):
                if j >= 0 and j < lengthEachSentence:
                    currentWord = eachSentence[j].lower()
                    if currentWord in dictionaryIndex:
                        wordIndex = dictionaryIndex[currentWord.lower()]
                    else:
                        wordIndex= 0
                    wordWithContext = wordWithContext + [wordIndex]
                elif j < 0:
                    wordWithContext = wordWithContext + [indexStartToken]
                elif j == lengthEachSentence:
                    wordWithContext = wordWithContext + [indexEndToken]
            listEachSentenceWindow = listEachSentenceWindow + [wordWithContext]
        listListWindows = listListWindows + [listEachSentenceWindow]  
    
    # Unpack the nested list into a single list. This will facilitate testing of output of this function.
    listWindows = []
    for listItem in listListWindows:
        listWindows = listWindows + listItem
    # Test if the count of word windows matches the count of labels
    if len(listWindows) != len(listAllLabels):
        print "Error in dataset: Mismatch between the number of word windows and number of labels."
    else:
        print "Total number of word windows equals the number of labels."

    return listListWindows, listWindows
