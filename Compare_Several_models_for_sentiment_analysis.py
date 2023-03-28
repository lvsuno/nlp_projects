import pandas as pd
import numpy as np
import spacy
import preprocess_lvsuno as pl
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier


def get_vec(w):
    """
    param w: sentences or an ensemble of sentence
    :return: a word vector
    """
    do = nlp(w)
    vec = do.vector
    return vec


nlp = spacy.load('en_core_web_lg')

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


# transform sentences to vector
dt['vec'] = dt['reviews'].apply(lambda x1: get_vec(x1))
X = dt['vec'].to_numpy()
X = X.reshape(-1, 1)
X = np.concatenate(np.concatenate(X, axis=0), axis=0).reshape(-1, 300)
y = dt['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)