import csv
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse as sp


###################
### IMPORT DATA ###
###################

# Import training data
train_label = []
train_data = []
with open('reviews_tr.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        if row[0] != "label":
            train_label.append(float(row[0]))
            train_data.append(row[1])


# Import test data
test_label = []
test_data = []
with open('reviews_te.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        if row[0] != "label":
            test_label.append(float(row[0]))
            test_data.append(row[1])


##########################
### CREATE WORD VECTOR ###
##########################

# Unigarm matrix
unigram_vectorizer = CountVectorizer(token_pattern = r"(?u)\b\w+\b")
unigram_x = unigram_vectorizer.fit_transform(train_data)
unigram_test = unigram_vectorizer.transform(test_data)
unigram_words = unigram_vectorizer.get_feature_names()+['INTERCEPT']


# TFIDF matrix
def vectorizeTFIDF(x):
    df = np.bincount(x.indices, minlength=x.shape[1])
    droplist = np.where(df == 0)[0]
    df = np.clip(df, a_min=1,a_max=float('inf'))
    idf = np.log10(float(x.shape[0]) / df)
    idf_diag = sp.spdiags(idf, diags=0, m=x.shape[1],n=x.shape[1], format='csr')
    return (x @ idf_diag,droplist)

tfidf_x = vectorizeTFIDF(unigram_x)[0]
tfidf_test,tfidf_test_droplist = vectorizeTFIDF(unigram_test)
tfidf_words = unigram_vectorizer.get_feature_names()+['INTERCEPT']


# Bigram matrix
bigram_vectorizer = CountVectorizer(token_pattern = r"(?u)\b\w+\b",ngram_range=(2,2))
bigram_x = bigram_vectorizer.fit_transform(train_data)
bigram_test = bigram_vectorizer.transform(test_data)
bigram_words = bigram_vectorizer.get_feature_names()+['INTERCEPT']

# Longgram matrix
longgram_vectorizer = CountVectorizer(token_pattern = r"(?u)\b\w\w\w+\b")
longgram_x = longgram_vectorizer.fit_transform(train_data)
longgram_test = longgram_vectorizer.transform(test_data)
longgram_words = longgram_vectorizer.get_feature_names()+['INTERCEPT']


#####################
### ADD INTERCEPT ###
#####################

af_unigram_x = scipy.sparse.hstack((unigram_x,np.ones(shape=(unigram_x.shape[0],1))),format="csr")
af_unigram_test = scipy.sparse.hstack((unigram_test,np.ones(shape=(unigram_test.shape[0],1))),format="csr")

af_tfidf_x = scipy.sparse.hstack((tfidf_x,np.ones(shape=(tfidf_x.shape[0],1))),format="csr")
af_tfidf_test = scipy.sparse.hstack((tfidf_test,np.ones(shape=(tfidf_test.shape[0],1))),format="csr")

af_bigram_x = scipy.sparse.hstack((bigram_x,np.ones(shape=(bigram_x.shape[0],1))),format="csr")
af_bigram_test = scipy.sparse.hstack((bigram_test,np.ones(shape=(bigram_test.shape[0],1))),format="csr")

af_longgram_x = scipy.sparse.hstack((longgram_x,np.ones(shape=(longgram_x.shape[0],1))),format="csr")
af_longgram_test = scipy.sparse.hstack((longgram_test,np.ones(shape=(longgram_test.shape[0],1))),format="csr")


########################
### TRAIN PERCEPTRON ###
########################

def trainPerceptron(x_matrix,y_matrix,n="all"):
    w_hat=np.zeros(x_matrix.shape[1])
    w=w_hat.copy()

    # Set a random order
    if n=="all":
        n=x_matrix.shape[0]
        randomorder = list(np.random.permutation(n)) + list(np.random.permutation(n))
    else:
        randomorder = list(np.random.permutation(n))


    for i in tqdm(range(len(randomorder))):
        x = x_matrix[randomorder[i]]
        y = y_matrix[randomorder[i]]*2-1

        if y * x.dot(w)[0] <= 0:
            w = np.ravel(w + x.multiply(y))

        if i>=n-1:
            w_hat += w

    return w_hat/(n+1)


unigram_w = trainPerceptron(af_unigram_x,train_label,n="all")
tfidf_w = trainPerceptron(af_tfidf_x,train_label,n="all")
bigram_w = trainPerceptron(af_bigram_x,train_label,n="all")
longgram_w = trainPerceptron(af_longgram_x,train_label,n="all")


#######################
### CALCULATE ERROR ###
#######################

def calculateError(w,x,y,droplist=[]):
    yhat = w @ x.T

    # For calculating training risk for TFIDF, do not count observations where df(t,D)=0
    if len(droplist)>0:
        indices = set(range(len(test_label)))-set(droplist)
        total = x.shape[0]-len(droplist)
    else:
        indices = range(x.shape[0])
        total = x.shape[0]

    return len([i for i in indices if yhat[i]*(y[i]*2-1) < 0]) / total

# Train Error
unigram_train_error = calculateError(unigram_w,af_unigram_x,train_label)
tfidf_train_error = calculateError(tfidf_w,af_tfidf_x,train_label)
bigram_train_error = calculateError(bigram_w,af_bigram_x,train_label)
longgram_train_error = calculateError(longgram_w,af_longgram_x,train_label)

# Test Error
unigram_test_error = calculateError(unigram_w,af_unigram_test,test_label,droplist=tfidf_test_droplist)
tfidf_test_error = calculateError(tfidf_w,af_tfidf_test,test_label)
bigram_test_error = calculateError(bigram_w,af_bigram_test,test_label)
longgram_test_error = calculateError(longgram_w,af_longgram_test,test_label)


###########################################
### IDENTIFY WORDS WITH EXTREME WEIGHTS ###
###########################################

def extremeWords(words,w,n=5):
    top5dict={}
    bottom5dict={}

    # Cycle through only the rows non-zero values
    for key, value in sp.dok_matrix(sp.csr_matrix(w)).items():
        if len(top5dict)<n:
            top5dict[key[1]]=value
        else:
            if value>=min(top5dict.values()):
                item = min(top5dict, key=top5dict.get)
                del top5dict[item]
                top5dict[key[1]] = value

        if len(bottom5dict)<n:
            bottom5dict[key[1]]=value
        else:
            if value<=max(bottom5dict.values()):
                item = max(bottom5dict, key=bottom5dict.get)
                del bottom5dict[item]
                bottom5dict[key[1]] = value

    return (list(np.take(words,sorted(top5dict))),
            list(np.take(words, sorted(bottom5dict))))


unigram_top_words = extremeWords(unigram_words,unigram_w,n=10)
tfidf_top_words = extremeWords(tfidf_words,tfidf_w,n=10)
bigram_top_words = extremeWords(bigram_words,bigram_w,n=10)
longgram_top_words = extremeWords(longgram_words,longgram_w,n=10)



#####################
### PRINT RESULTS ###
#####################

# Error Rates
print("Unigram train error: {:.3f}".format(unigram_train_error))
print("Unigram test error: {:.3f}".format(unigram_test_error))
print("TFIDF train error: {:.3f}".format(tfidf_train_error))
print("TFIDF test error: {:.3f}".format(tfidf_test_error))
print("Bigram train error: {:.3f}".format(bigram_train_error))
print("Bigram test error: {:.3f}".format(bigram_test_error))
print("Longgram train error: {:.3f}".format(longgram_train_error))
print("Longgram test error: {:.3f}".format(longgram_test_error))


# Top Words
print("  {:<25} {:<25} {:<25} {:<25}".format('Unigram', 'TFIDF', 'Bigram', 'Longgram'))
print("-" * 90)
print("Top 10 Words:")
for i in range(10):
    print("  {:<25} {:<25} {:<25} {:<25}".format(unigram_top_words[0][i],
                                                tfidf_top_words[0][i],
                                                bigram_top_words[0][i],
                                                longgram_top_words[0][i]))
print("\nBottom 10 Words:")
for i in range(10):
    print("  {:<25} {:<25} {:<25} {:<25}".format(unigram_top_words[1][i],
                                                tfidf_top_words[1][i],
                                                bigram_top_words[1][i],
                                                longgram_top_words[1][i]))
