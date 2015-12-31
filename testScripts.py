# Test scripts for the function generateWordWindows(). File generated is in the output folder. 

with open("./output/testGenerateWordWindows.txt", "w") as textFileObj:
    for i in range(11):
        textFileObj.write(str(all_sentences[i]) + '\n')
        textFileObj.write(str(nestedListSentenceTuples[i]) + '\n')
        for j in range(len(all_sentences[i])):
            try:
                textFileObj.write(str(all_sentences[i][j].lower()) + '\n')
                textFileObj.write(str(indexedVocabulary[all_sentences[i][j].lower()]) + '\n')
            except:
                continue  
