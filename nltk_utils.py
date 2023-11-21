import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer=PorterStemmer()

def tokenize(sentence):
    return word_tokenize(sentence)
def stem(word):
    return stemmer.stem(word)
def bag_of_words(pattern_sentence,all_words):
    pattern_sentence=[stem(w) for w in pattern_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for index,w in enumerate(all_words):
        if w in pattern_sentence:
            bag[index]=1.0
    return bag