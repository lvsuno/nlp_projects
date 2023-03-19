import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
'''
 preprocessing package build by myself
 https://github.com/lvsuno/Text-preprocessing
'''
import preprocess_lvsuno as pl



# from matplotlib.animation import FuncAnimation


# read the dataset
df = pd.read_csv('data/twitt30k.csv')
# count the number of observations
print(df['sentiment'].value_counts())




# define a function that train and evaluate svm classifier
def run_svm(df1):
    X = df1['twitts']
    y = df1['sentiment']
    print(y.unique())
    # Different results can be observed with different hyperparameters
    tfidf = TfidfVectorizer(norm='l1', ngram_range=(1, 2), analyzer='word', max_features=5000)
    X = tfidf.fit_transform(X)
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    print('shape of X: ', X.shape)
    # train svm classifier
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    # Predict
    y_pred = svc.predict(X_test)

    print('Printing Report')
    print(classification_report(y_test, y_pred))

    return tfidf, svc


tfidf, clf_svc = run_svm(df)
# Predict a new twitter sentiment
x1 = ['i am really happy. thanks a lot for coming with me']
print(clf_svc.predict(tfidf.transform(x1)))

# plt.style.use('fivethirtyeight')
# frame_len = 10000


## Preprocess the data before training the classifier

# Put twitts into the lower case
df['twitts'] = df['twitts'].apply(lambda x : x.lower())
# Expand any contracted words
df['twitts'] = df['twitts'].apply(lambda x : pl.cont_exp(x))

# remove emails, urls, rt, html tags, special characters

df['twitts'] = df['twitts'].apply(lambda x: pl.remove_emails(x))
df['twitts'] = df['twitts'].apply(lambda x: pl.remove_urls(x))
df['twitts'] = df['twitts'].apply(lambda x: pl.remove_rt(x))
df['twitts'] = df['twitts'].apply(lambda x: pl.remove_html_tags(x))
df['twitts'] = df['twitts'].apply(lambda x: pl.remove_special_chars(x))

tfidf_pre, clf_svc_pre = run_svm(df)
tfidf, clf_svc = run_svm(df)
# Predict a new twitter sentiment
x1 = ['i am really happy. thanks a lot for coming with me']
print(clf_svc.predict(tfidf.transform(x1)))