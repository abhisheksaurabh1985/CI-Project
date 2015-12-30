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
        

def _split_into_sentences(trainingDataFileName):
    """
    Extract the labels and turn the words into sentences

    :return:
    """
    EOL_CHARS = ".!?"
    allSentences = []
    labels = []
    with open(trainingDataFileName, 'r') as f:
        sentence = []
        for line in f:
            try:
                print line
                word, label = line.split()
            except:
                continue
            #no the end of the sentence
            sentence += [word]
            intLabel = 1 if label == "PERSON" else 0
            labels += [intLabel]
            if word in EOL_CHARS:
                allSentences += [sentence]
                sentence = []

        #in case the last sentence doesn't end with proper punctuation!
        if sentence != []:
            allSentences += [sentence]
##        self.allSentences = allSentences
        return allSentences, labels    
