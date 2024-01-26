import pickle
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import string
import sys

def getCategory(newTicket):
    dataDict = {0 :'Lost Accessories', 1:'Operating System', 2:'Security Incident', 3:'Login issues', 4:'Application Issues', 5:'Display Configuration Issues', 6:'Hardware Issues', 7:'Reconciliation Activity', 8:'Hosted Applications', 9:'Video Conference', 10:'Network Issues', 11:'IP Phone', 12:'UAT or Testing'}
    
    #load count_vect
    #count_vect = CountVectorizer()
    count_vect = pickle.load(open("C:\Git\hackday\Engines\count_vect.pickle", "rb"))
    
    #load transformer
    transformer = pickle.load(open("C:\\Git\\hackday\\Engines\\transformer.pickle" ,"rb"))
    
    
    #load model from disk
    multinomialNB_filename = 'C:\Git\hackday\Engines\multinomialNB_classifier.pkl'
    loaded_model = pickle.load(open(multinomialNB_filename, 'rb'))
    
    #newTicket = "Laptop is not working"
    ticket = [newTicket]
    testCount = count_vect.transform(ticket)
    testInvFrecounts = transformer.transform(testCount)
    testPredicted = loaded_model.predict(testInvFrecounts)
    print(dataDict[int(testPredicted)])
    return dataDict[int(testPredicted)]
    