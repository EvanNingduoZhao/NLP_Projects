import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
import time


wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# use beautiful package to open XMl file and only take the contents with 'review_text' tag
positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.find_all('review_text')

# do the same thing with the negative reviews
negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.find_all('review_text')

# randomly shuffle the positive reviews and only keep the same number of positive reviews as that of the negative ones
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

def my_tokenizer(s):
    #change all words to lowercase
    s=s.text.lower()
    #tokenize
    tokens = nltk.tokenize.word_tokenize(s)
    #only keep the words longer than 2 letter
    tokens = [t for t in tokens if len(t)>2]
    #lemmatize all words
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    #remove stop words
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

#word_index_map is a dict with tokens as key and its index as value, each token's index is its position in the feature vector
word_index_map ={}
current_index=0

# the following two are list of lists, positive_tokenized consists of a bunch of lists, each list within is consist of
# all the tokenized tokens of one positive review
positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review)
    #each time we append, we append a list, this list is consisted of tokenized tokens of one positive review
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index +=1

for review in negative_reviews:
    tokens = my_tokenizer(review)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index +=1

#note for the positive and negative reviews, we used the same word_index_map. Therefore, now the len of word_index_map
# is equal to the number of unique words that exist in the whole dataset, of course after removing stop words and lemmatization

# now we are ready to make an list of normalized counts for each review,
# of course the len of each list will be the len of word_index_map, n
# and the number of such list is equal to the total number of reviews in the dataset, N

n = len(word_index_map)
N = len(positive_tokenized)+len(negative_tokenized)


# input parameters of this method is tokens (list): a list of tokenized tokens of one review (an element in postive_tokenized)
#                                    laben  (int): if this review is a positive (1) one or a negative (0) one
# the method returns a feature vector of a review consisting of normalized counts
def tokens_to_vector(tokens,label):
    # we make the size of x as n+1 since we want to add label (pos or neg )to the end of each x
    x = np.zeros(n+1)
    for t in tokens:
        i = word_index_map[t]
        x[i]=+1
    x=x/x.sum()
    x[-1]=label
    return x


# we use matrix to store all the feature vectors
matrix = np.zeros((N,n+1))
i =0

for tokens in positive_tokenized:
    #pass the second parameter as 1 since we are dealing with positive reviews
    xy = tokens_to_vector(tokens,1)
    # each row in matrix is a feature vector for a review, assign xy to the corresponding row
    matrix[i] = xy
    i+=1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens,0)
    matrix[i] = xy
    i+=1


np.random.shuffle(matrix)

# assign the normalized counts feature vectors to X
X = matrix[:,:-1]
# assign the pos or neg binary labels to Y
Y = matrix[:,-1]

# split X and Y to train and test data
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2)

print('Statistics of RandomForest model')
start_time = time.time()
model = RandomForestClassifier()
model.fit(Xtrain,Ytrain)

#note the model.predict method requires an input of a 2D array, so if you are just passing in one feature vector x
#pass it like model.preiction([x])
prediction = model.predict(Xtest)
print("--- %s seconds ---" % (time.time() - start_time))

# precision:在所有被predict为positive的reviews里有百分之多少真的是positive的 TP/(TP+FP)
# recall: 在所有真的是positive的reviews里面，有百分之多少被predict成positive了 TP/(TP+FN)
# f1_score: weighted harmonic mean of precision and recall, F1 Score = 2*(Recall * Precision) / (Recall + Precision)
# support: test set 里面真的是postive或者negative的有多少个
print(classification_report(Ytest,prediction))

# in a confusion matrix, diagonal entries represent 这个class里面被label对的有多少个，rows 代表true labes，
# cols 代表predicted labels, 因此off diagonal entries可以以此标准判断是怎么回事
print(confusion_matrix(Ytest,prediction))


# show all the words that have a weight with absolute value larger than threshold
# threshold = 0.5
# for word, index in word_index_map.items():
#     weight = model.coef_[0][index]
#     if weight>threshold or weight < -threshold:
#         print(word,weight)


# the following is just a play_around code to test the model's prediction against specific reviews
# k = 0
# for review in negative_reviews[:20]:
#     print(review.text)
#     review = my_tokenizer(review)
#     x = np.zeros(n)
#     for t in review:
#         i = word_index_map[t]
#         x[i] = +1
#     x = x / x.sum()
#     result = model.predict([x])
#     print(result)
#     print()
#     print()
#     print()
#     print()
#     test to code with this beast