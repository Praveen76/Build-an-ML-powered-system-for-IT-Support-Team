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


'''
#it will print the encoding of the csv use it in encoding inside read_csv
with open('stage1.csv') as f:
   print(f)
'''


#preprocessing
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def preProcessing(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

df = pd.read_csv("C:\Git\hackday\stage1.csv", header=None,
                   names=['label', 'message'], encoding='cp1252')

df = df.dropna(axis = 0, how ='any')

dataDict = {0 :'Lost Accessories', 1:'Operating System', 2:'Security Incident', 3:'Login issues', 4:'Application Issues', 5:'Display Configuration Issues', 6:'Hardware Issues', 7:'Reconciliation Activity', 8:'Hosted Applications', 9:'Video Conference', 10:'Network Issues', 11:'IP Phone', 12:'UAT or Testing'}

df['label'] = df.label.map({'Lost Accessories': 0, 'Operating System': 1, 'Security Incident': 2, 'Login issues': 3, 'Application Issues': 4, 'Display Configuration Issues': 5, 'Hardware Issues': 6, 'Reconciliation Activity':7, 'Hosted Applications':8, 'Video Conference': 9, 'Network Issues': 10, 'IP Phone': 11, 'UAT or Testing' : 12}) 

df['message'] = df.message.map(lambda x: x.lower()) 

df['message'] = [ preProcessing(doc).split() for doc in df['message']]

#pre processing ends
df['message'] = df['message'].apply(lambda x: ' '.join(x))

print(df)

count_vect = CountVectorizer()
count_vect.fit(df['message'])
rep1 = count_vect.vocabulary_
counts = count_vect.transform(df['message'])
countsArray = counts.toarray()

#store count_vect
pickle.dump(count_vect, open("count_vect.pickle", "wb"))


transformer = TfidfTransformer().fit(counts)
#store transformer
pickle.dump(transformer, open("transformer.pickle", "wb"))


InvFrecounts = transformer.transform(counts)  
countInverseFrequency = InvFrecounts.toarray()


X_train, X_test, y_train, y_test = train_test_split(InvFrecounts, df['label'], test_size=0.1, random_state=69)

model1 = MultinomialNB().fit(X_train, y_train)

predicted = model1.predict(X_test)

print("MultinomialNB prediction")
print(np.mean(predicted == y_test))
print(confusion_matrix(y_test, predicted))


ticket = ["Canada citrix issue"]
testCount = count_vect.transform(ticket)
testInvFrecounts = transformer.transform(testCount) 
testPredicted = model1.predict(testInvFrecounts)
print(dataDict[int(testPredicted)])

#save the model
multinomialNB_filename = 'multinomialNB_classifier.pkl'
pickle.dump(model1, open(multinomialNB_filename, 'wb'))



