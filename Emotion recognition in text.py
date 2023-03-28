import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import preprocess_lvsuno as pl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
import joblib as jl

# Read the dataset
dt = pd.read_csv('data/text_to_emotion.csv')
dt1 = dt.copy()
# Put the text in lowercase
dt['text'] = dt['text'].apply(lambda x: str(x).lower())
# Expand contractions
dt['text'] = dt['text'].apply(lambda x: pl.cont_exp(x))
# Remove special characters
dt['text'] = dt['text'].apply(lambda x: pl.remove_special_chars(x))
# Remove accented characters
dt['text'] = dt['text'].apply(lambda x: pl.remove_accented_chars(x))

# Load Glove Vector
glove_vectors = dict()

file = open('Glove/glove.6B.100d.txt', encoding='utf-8')

for line in file:
    values = line.split()
    word = values[0]
    vectors = np.asarray(values[1:])
    glove_vectors[word] = vectors

file.close()

# Text to Glove
glo_shape = 100


def get_vec(x):
    """
    To get a sentence vector, we have to find the mean of word's vector
    that it is composed of.
    :param x:
    :return:
    """
    arr = np.zeros(glo_shape)

    text = str(x).split()

    for t in text:
        try:
            vec = glove_vectors.get(t).astype(float)
            arr = arr + vec
        except:
            pass

    arr = arr.reshape(1, -1)[0]
    return arr / len(text)


dt['vec'] = dt['text'].apply(lambda x: get_vec(x))

# Get the predictors and the labels
X = dt['vec']
y = dt['emotion']

# Reshape the vector of each sentence
X = np.concatenate(X, axis=0).reshape(-1, glo_shape)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# Define the classifier
clf = LogisticRegression(solver='liblinear', multi_class='auto')
# fit the classifier
clf.fit(X_train, y_train)
# Predict X_test
y_pred = clf.predict(X_test)
print('########## Classification report Logistic Regression ##########')
print(classification_report(y_test, y_pred))

print('########## Confusion matrix Logistic Regression ##########')
print(confusion_matrix(y_test, y_pred))

##################### Using SVM #######################
clf = LinearSVC()
# fit the classifier
clf.fit(X_train, y_train)
# Predict X_test
y_pred = clf.predict(X_test)
print('########## Classification report SVM ##########')
print(classification_report(y_test, y_pred))

print('########## Confusion matrix SVM ##########')
print(confusion_matrix(y_test, y_pred))

'''
For more automatic manner, we can use Pipeline from sklearn package
'''


def pre_pro(x):
    x = str(x).lower()
    x = pl.cont_exp(x)
    x = pl.remove_special_chars(x)
    x = pl.remove_accented_chars(x)
    vec = get_vec(x)  # .reshape(-1, glo_shape)
    # if vec.shape[1] != glo_shape:
    #   print(x)
    return vec


def Pre_pro_full(X1):
    # print(type(X1))
    x1 = X1.apply(pre_pro)
    # x1 = x1.reshape(-1, glo_shape)
    x1 = x1.values.tolist()
    x1 = np.reshape(x1, (-1, glo_shape))
    return x1


pipe = Pipeline(steps=[('prep', FunctionTransformer(Pre_pro_full)), ('clf', LinearSVC())])

X_train, X_test, y_train, y_test = train_test_split(dt1['text'], dt1['emotion'], test_size=0.2, random_state=0,
                                                    stratify=dt1['emotion'])

pipe.fit(X_train, y_train)

yhat = pipe.predict(X_test)
print('########## Classification report SVM pipeline ##########')
print(classification_report(y_test, yhat))

print('########## Confusion matrix SVM pipeline ##########')
print(confusion_matrix(y_test, yhat))
with open("models/Emo_glo_gzip.joblib", 'wb') as f:
    jl.dump(pipe, f, compress='gzip')

x = 'Thanks a lot but i feel alone.'
yhat = pipe.predict(pd.Series([x]))
print(yhat)
