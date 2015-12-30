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
        

def getSentences(trainingDataFileName):

    # Define end-of line characters
    endOfLineCharacters = '.?!'
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

