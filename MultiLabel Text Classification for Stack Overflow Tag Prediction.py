import pandas as pd
import numpy as np
import ast
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import joblib as jl




def jacc_similarity(y_true, y_pred):
    """
    Jaccard similarity: intersection(A,B)/union(A,B)
    :param y_true: True label
    :param y_pred: Predicted label
    :return:
    """
    jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(axis=1)
    return jaccard.mean() * 100

# read the data
'''
The original data can be find here: https://www.kaggle.com/datasets/stackoverflow/stacksample
The top 20 tags used is selected and the text data is preprocessed:
lower case, expand contractions, remove url and email, remove html tags, remove accented characters
remove special characters and character repetiton
'''
df = pd.read_csv('data/stackoverflow.csv', index_col=0)
# Print the columns
print(df.columns)
# Remove Na
df = df.dropna()

print(type(df.iloc[0]['Tags']))
# Convert string to list (tags in the file are string)
df['Tags'] = df['Tags'].apply(lambda x: ast.literal_eval(x))

multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df['Tags'])
classes = multilabel.classes_
print(classes)

# Obtain TfIdf vector
tfidf = TfidfVectorizer(analyzer='word', max_features=1000, ngram_range=(1, 3), stop_words='english')

X = tfidf.fit_transform(df['Text'])
# print(tfidf.vocabulary_)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Logistic Regression wih One vs Rest strategy
lr = LogisticRegression(solver='lbfgs')
clf = OneVsRestClassifier(lr)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('############Logistic Regression##########')
print(jacc_similarity(y_test, y_pred))

###### SVM ############

svm = LinearSVC(C=1.5, penalty='l1', dual=False)
clf = OneVsRestClassifier(svm)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('############ SVM ##########')
print(jacc_similarity(y_test, y_pred))

x = ['how to write ml code in python and java i have data but do not know how to do it']
xt = tfidf.transform(x)
print(clf.predict(xt))
# Reverse the label to give the true label
print(multilabel.inverse_transform(clf.predict(xt)))

with open('models/Multilabel_Svm.joblib', 'wb') as f:
    jl.dump(clf, f, compress='gzip')

with open('models/Multilabel_tfidf.joblib', 'wb') as f:
    jl.dump(tfidf, f, compress='gzip')