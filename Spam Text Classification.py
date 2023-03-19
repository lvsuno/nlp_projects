import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/spam.tsv', sep='\t')

# show the five first rows of the data
print(df.head())
# check if there's some null values
print(df.isnull().sum())

# check if there's some NA values
print(df.isna().sum())
# get the numbers of observations and the number of
rows, cols = df.shape
# See how many observations we've per category
print(df['label'].value_counts())
# Balance Dataset
ham = df[df['label'] == 'ham']
spam = df[df['label'] == 'spam']
ham = ham.sample(spam.shape[0])
dt = pd.concat([ham, spam], ignore_index=True)
print(dt['label'].value_counts())

## Explore the data

'''
# plot  the histogram of 'length'
plt.hist(ham['length'], bins=100, alpha=0.5, label='Ham')
plt.hist(spam['length'], bins=100, alpha=0.5, label='Spam')
plt.legend()
plt.savefig('images/spam_length_hist.jpg')

plt.figure()
# plot  the histogram of 'punct'
plt.hist(ham['punct'], bins=100, alpha=0.5, label='Ham')
plt.hist(spam['punct'], bins=100, alpha=0.5, label='Spam')
plt.legend()
plt.savefig('images/spam_punct_hist.jpg')
plt.show()
'''
# Appply TfIDF to the data
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(dt['message'])
X = X.toarray()

# split in Training and Test
X_train, X_test, y_train, y_test = train_test_split(X, dt['label'], test_size=0.3, random_state=1, stratify=dt['label'])


# models
# get a list of models to evaluate
def get_models():
    models = dict()
    # define the models
    models['RFC'] = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    models['SVC'] = SVC(C=1000, gamma='auto')
    return models


def evaluate_model(mo, X1, y1, x):
    mo.fit(X1, y1)
    pred = mo.predict(x)
    return pred


# To predict new value...
# We can combine all with pipeline
def predict(mod, x):
    x = tfidf.transform([x])
    x = x.toarray()
    pred = mod.predict(x)
    return pred


# evaluate models
results, names = list(), list()
models = get_models()
for name, model in models.items():
    y_pred = evaluate_model(model, X_train, y_train, X_test)
    results.append(y_pred)
    names.append(name)
    print('>%s' % name)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(predict(model, 'hey, how are you?'))
    print(predict(model, 'you have got free visa for canada, no hidden fees. Please contact me'))
