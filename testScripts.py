# Test scripts for the function generateWordWindows()

# Get first sentence
all_sentences[0]

# Get word tuples for the first sentence
for i in range(len(all_sentences[0])):
    print sentenceTuples[i]

# Get index from the indexedVocabulary    
for i in range(len(all_sentences[0])):
    try:
        print all_sentences[0][i].lower(), indexedVocabulary[all_sentences[0][i].lower()]
    except:
        continue
