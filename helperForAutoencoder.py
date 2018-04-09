import nltk
import numpy as np


def getLookUps (path):
    #input: path of dataset
    #output: two lookups- 1 gives the index corresponding to each word in the vocabulry
    #and other which gives the word corresponding to the index
    file_content = open(path).read()
    sentences = nltk.tokenize.sent_tokenize(file_content)
    vocab = {}
    tokens = {}

    for a in sentences:
        tokens = nltk.word_tokenize(a)
        #del tokens[0]
        index=0
        for k in tokens:
            vocab[k] = 1
    vocab['EOS']=1
    vocab['PAD']=0
    index=2
    for key, value in vocab.items():
        vocab[key]=index
        index+=1
    reverseLookUp ={}

    for key, value in vocab.items():
        reverseLookUp[value]=key

    return vocab, reverseLookUp

def wordEmbeddings(sentence, lookUp):
    words = nltk.word_tokenize(sentence)
    embeddings =[]
    for w in words:
        embeddings.append(lookUp[w])
    return embeddings

def createBatch(sentences, startIndex, batchSize):
    if startIndex+batchSize<len(sentences):
        return sentences[startIndex:startIndex+batchSize]
    else:
        return sentences[startIndex:]
#a=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#print(createBatch(a,7,5))
def getSentences(path,vocab):
    file_content = open(path).read()
    sentences = nltk.tokenize.sent_tokenize(file_content)
    tokens = {}
    tokenList= []
    for a in sentences:
        tokens = nltk.word_tokenize(a)
        b= []
        #del tokens[0]
        for a in tokens :
            b.append(vocab[a])

        tokenList.append(b)
    return tokenList

def toWords(input, reverseLookUp):
    wordSequence =[]
    for element in input:
        if element in reverseLookUp:
            wordSequence.append(reverseLookUp[element])
        else:
            wordSequence.append('.')
    return wordSequence