import gzip

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocess_lvsuno as pl
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from pickle import dump, load
import spacy



def get_vec(w):
    """
    param w: sentences or an ensemble of sentence
    :return: a word vector
    """
    do = nlp(w)
    vec = do.vector
    return vec


nlp = spacy.load('en_core_web_lg')

# Read the data

dt = pd.read_csv('data/imdb_reviews.txt', sep='\t', header=None)
dt.columns = ['reviews', 'sentiment']

# Apply preprocessing step
dt['reviews'] = dt['reviews'].apply(lambda x: pl.cont_exp(x))
dt['reviews'] = dt['reviews'].apply(lambda x: pl.remove_special_chars(x))
dt['reviews'] = dt['reviews'].apply(lambda x: pl.remove_accented_chars(x))
dt['reviews'] = dt['reviews'].apply(lambda x: pl.remove_emails(x))
dt['reviews'] = dt['reviews'].apply(lambda x: pl.remove_html_tags(x))
dt['reviews'] = dt['reviews'].apply(lambda x: pl.remove_urls(x))
dt['reviews'] = dt['reviews'].apply(lambda x: pl.make_base(x))
dt['reviews'] = dt['reviews'].apply(lambda x: str(x).lower())
dt['reviews'] = dt['reviews'].apply(lambda x: pl.spelling_correction(x).raw_sentences[0])

X = dt['reviews']
y = dt['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

pipe = Pipeline(steps=[('tfidf', TfidfVectorizer()),
                       ('clf', LogisticRegression(solver='liblinear'))])

hyperparameters = {
    'tfidf__max_df': (0.5, 1.0),
    'tfidf__ngram_range': ((1, 1), (1, 2)),
    'tfidf__use_idf': (True, False),
    'tfidf__analyzer': ('word', 'char', 'char_wb'),
    'clf__penalty': ('l2', 'l1'),
    'clf__C': (1, 2)
}

clf = GridSearchCV(pipe, hyperparameters, n_jobs=-1, cv=None)
clf.fit(X_train, y_train)

print(clf.best_estimator_, clf.best_params_, clf.best_score_)

y_pred = clf.predict(X_test)
print('######### Results with Logistic Regression #######')
print(classification_report(y_test, y_pred))

# With SVM
pipe = Pipeline(steps=[
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

hyperparameters = {
    'tfidf__max_df': (0.5, 1.0),
    'tfidf__ngram_range': ((1, 1), (1, 2)),
    'tfidf__use_idf': (True, False),
    'tfidf__analyzer': ('word', 'char', 'char_wb'),
    'clf__C': (1, 2, 2.5, 3)
}

clf = GridSearchCV(pipe, hyperparameters, n_jobs=-1, cv=5)
clf.fit(X_train, y_train)

print(clf.best_estimator_, clf.best_params_, clf.best_score_)

y_pred = clf.predict(X_test)
print('######### Results with SVM #######')
print(classification_report(y_test, y_pred))

dump(clf, open("models/senti_svm.pkl", "wb"))

# Save model as gz file in binary data
'''
Saving and loading files (with joblib) like the following codes, doesn't load the pipeline correctly
However it works with pickle but this later doesn't allow compression
'''

# joblib.dump(pipe, gzip.open('models/senti_svm.dat.gz', "wb"))
# model = joblib.load('models/senti_svm.dat.gz', mmap_mode='rb')
# model = load(open("models/senti_svm.pkl", "rb"))

with open("models/senti_svm_gzip.joblib", 'wb') as f:
    joblib.dump(clf, f, compress='gzip')

with open("models/senti_svm_gzip.joblib", 'rb') as f:
    model = joblib.load(f)

print(model.predict(['i have watched this movie. plot is straight. return my money']))

# Using Word2Vec embedding for Sentiment Analysis
'''
x = 'dog cat lion dsfaf'
doc = nlp(x)

for token in doc:
    print(token.text, token.has_vector, token.vector_norm)

# Check the similarity
for token1 in doc:
    for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))
'''
# transform sentences to vector
dt['vec'] = dt['reviews'].apply(lambda x1: get_vec(x1))
X = dt['vec'].to_numpy()
X = X.reshape(-1, 1)
X = np.concatenate(np.concatenate(X, axis=0), axis=0).reshape(-1, 300)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

with open("models/senti_w2v_gzip.joblib", 'wb') as f:
    joblib.dump(clf, f, compress='gzip')


######### Word2vec with svm #############
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

log = LogisticRegression(solver='liblinear')

hyperparameters = {
    'penalty': ['l1', 'l2'],
    'C': (1, 2, 3, 4)
}

clf = GridSearchCV(log, hyperparameters, n_jobs=-1, cv=5)
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))