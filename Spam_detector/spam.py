from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('spambase.data').as_matrix()
# print(data)


# with pd.option_context('display.max_rows',None,'display.max_columns',None):
#     print(data)
np.random.shuffle(data) # we want to randomize the order of rows and then split them into train and test sets

X=data[:,:48]
# print(X.shape)
Y=data[:,-1]
# print(Y)

Xtrain = X[:-100,]
# print(Xtrain)
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain,Ytrain)

print("Classification rate for Naive Bayes:", model.score(Xtest,Ytest))


model = AdaBoostClassifier()
model.fit(Xtrain,Ytrain)

print("Classification rate for Adaboost:",model.score(Xtest,Ytest))
