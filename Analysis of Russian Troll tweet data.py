import re
# from pickle import load, dump

import matplotlib
from nltk import WordNetLemmatizer

print('Default backend: ' + matplotlib.get_backend())
matplotlib.use("module://mplcairo.macosx")
print('Backend is now ' + matplotlib.get_backend())
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preprocess_lvsuno as pl
from PIL import Image
from beautifultable import BeautifulTable
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import advertools as adv
import decorators_for_data_science as dc
from matplotlib.font_manager import FontProperties

# Load Apple Color Emoji font
prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')

# nltk.download('omw-1.4')


# Read file where the duplicated tweets are removed
filename = 'data/russian-troll-tweets-master/Tweets_eng_remove_dup.csv'
df = pd.read_csv(filename, dtype={"tweet_id": float, "content": "string", "content_pre": "category"})
# Be sure that tweets are string (Errors can occur later)
df['content'] = df['content'].astype(str)
# columns name
print(df.columns)
# Dimension of the data
print(df.shape)

'''
Drop Na if exist and save it for later use. In this actual version of this code
has already applied it
'''


# select row with na
# dt = df[df.isna().any(axis=1)]
# print(dt['content'])
# df = df.dropna()
# df.to_csv(filename, index=False)


# Preprocessing
def prepro(text):
    # Put tweets into the lower case
    text = text.lower()
    # Expand any contracted words
    text = pl.cont_exp(text)

    # remove emails
    text = pl.remove_emails(text)
    # remove urls
    text = pl.remove_urls(text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # remove retweets
    text = pl.remove_rt(text)
    # removal html tags
    text = pl.remove_html_tags(text)
    # remove special characters
    text = pl.remove_special_chars(text)
    # Remove multiple space characters
    text = re.sub('\s+', ' ', text)
    # Remove new line characters
    text = re.sub('[\r\n]+', ' ', text)
    return text


# Get Top n Big N-grams
@dc.timing_decorator
@dc.log_execution
def get_top_n_Ngram(corpus, n=None, ngram_range=(1, 1), verbose=True):
    """
    Take in entry the following parameters
    :param verbose: print or not the table of the word and their frequency
    :param corpus: The corpus
    :param n: The number of top Ngrams
    :param ngram_range: a tuple of ngram (1, 2) means one to 2 gram
    :return: word frequency
    """
    Vect = CountVectorizer(ngram_range=ngram_range).fit(corpus)
    Bag_of_words = Vect.transform(corpus)
    Sum_words = Bag_of_words.sum(axis=0)
    Words_freq = [(word, Sum_words[0, idx]) for word, idx in Vect.vocabulary_.items()]
    Words_freq = sorted(Words_freq, key=lambda x: x[1], reverse=True)
    common_words = Words_freq[:n]
    if verbose:
        table = BeautifulTable()
        table.columns.header = ['Words', 'Freq']
        for word, freq in common_words:
            table.rows.append([word, freq])
        print(table)
        df1 = pd.DataFrame(common_words, columns=['NGram', 'Count'])
        df1 = df1.groupby('NGram').sum()['Count'].sort_values(ascending=False)

    return df1


@dc.timing_decorator
@dc.log_execution
def create_word_cloud(word, mask_image, stop_w, dest_file, col=True):
    """

    :param word: list of word
    :param mask_image: path to the mask
    :param stop_w: list of stopword
    :param dest_file: destination file
    :param col: collocations is true of false
    :return:
    """
    mask = np.array(Image.open(mask_image).convert("RGB"))
    mask[mask.sum(axis=2) == 0] = 255
    if stop_w is None:
        stpwd = stop_w
    else:
        stpwd = set(stop_w)

    word_cloud = WordCloud(contour_width=5,
                           background_color="white",
                           # mode="RGBA",
                           stopwords=stpwd,
                           mask=mask,
                           collocations=col,
                           color_func=ImageColorGenerator(mask)).generate(word)
    # Create coloring from the image
    plt.axis('off')
    plt.tight_layout(pad=0)
    # word_cloud.recolor(color_func=img_colors)
    plt.imshow(word_cloud, interpolation="bilinear")

    # Store the image
    plt.savefig(dest_file, format="png")
    plt.show()


@dc.timing_decorator
@dc.log_execution
def get_mentions_list(dat):
    """
    Get the mentions in a set of twitter's text
    :param dat: data frame column
    :return: list of mentions
    """
    # Retrieves all occurrences of @+text
    dat = dat.astype(str)
    col = dat.columns
    dat[col[0]] = dat[col[0]].str.findall(r'@\w+')
    # dat1 = dat.apply(lambda x: re.findall(r'@\w+', x))
    # Removes the @ in front
    dat[col[0]] = [list(map(lambda x: x[1:], mentions)) for mentions in dat[col[0]]]
    # Converts the list of words in each row to a string
    dat[col[0]] = dat[col[0]].apply(lambda x: ' '.join(x))
    # Concatenates all strings in one string
    all_mentions = ' '.join([word for word in dat[col[0]]])
    return all_mentions


@dc.timing_decorator
@dc.log_execution
def get_hashtag_list(dat):
    """
    Get the mentions in a set of twitter's text
    :param dat: data frame column
    :return: list of mentions
    """
    # Retrieves all occurrences of #+text
    dat = dat.astype(str)
    col = dat.columns
    dat[col[0]] = dat[col[0]].str.findall(r'#\w+')
    # dat1 = dat.apply(lambda x: re.findall(r'@\w+', x))
    # Removes the @ in front
    dat[col[0]] = [list(map(lambda x: x[1:], mentions)) for mentions in dat[col[0]]]
    # Converts the list of words in each row to a string
    dat[col[0]] = dat[col[0]].apply(lambda x: ' '.join(x))
    # Concatenates all strings in one string
    all_hashtags = ' '.join([word for word in dat[col[0]]])
    return all_hashtags


# Define a function to extract emoticons

@dc.timing_decorator
@dc.log_execution
def plot_top_emoticon(dat, n=20):
    """
    Get the emoji in a set of twitter's text
    :param n: Number of most frequent emoticons to plot
    :param dat: data series column
    :return: list of mentions
    """
    dat = dat.tolist()
    emoji_summary = adv.extract_emoji(dat)
    emoji_list = emoji_summary['top_emoji']
    # dump(emoji_summary, open('saved_variable/emoji_summary_troll_en.pkl', 'wb'))
    # emoji_summary = load(open('saved_variable/emoji_summary_troll_en.pkl', 'rb'))
    # emoji_list = emoji_summary['top_emoji']

    labels = list(list(zip(*emoji_list[0:n]))[0])
    count = list(list(zip(*emoji_list[0:n]))[1])
    p1 = plt.bar(np.arange(len(labels)), count, 0.8)
    plt.ylim(0, plt.ylim()[1] + 100)
    plt.xticks(np.arange(len(labels)+1))

    for rect, label in zip(p1, labels):
        height = rect.get_height()
        plt.annotate(label,
                     (rect.get_x() + rect.get_width() / 2, height + 5),
                     ha="center",
                     va="bottom",
                     fontsize=10,
                     fontproperties=prop
                     )
    plt.ylabel('Count')
    plt.xlabel('Emojis', fontname='Apple Color Emoji')
    plt.title('The top most used emojis')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.savefig('images/The_top_most_used_emojis.png')

    plt.figure()

    labels_1 = list(list(zip(*emoji_list[-n:]))[0])
    count_1 = list(list(zip(*emoji_list[-n:]))[1])
    p2 = plt.bar(np.arange(len(labels_1)), count_1, 0.8)
    plt.ylim(0, plt.ylim()[1]+1)
    plt.xticks(np.arange(len(labels)))

    for rect, label in zip(p2, labels_1):
        height = rect.get_height()
        plt.annotate(label,
                     (rect.get_x() + rect.get_width() / 2, height),
                     ha="center",
                     va="bottom",
                     fontsize=10,
                     fontproperties=prop
                     )
    plt.ylabel('Count')
    plt.xlabel('Emojis', fontname='Apple Color Emoji')

    plt.title('The least used emojis')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.savefig('images/The_least_used_emojis.png')

    plt.show()
    # return emoji_summary
    return emoji_list


'''
Apply preprocessing by suppressing some unuseful words. Remove Na if it occurs 
after the suppression and save the file for a later use. This version has already done it.
'''

df['content_pre'] = df['content'].apply(prepro)
df = df.dropna()
print(df.sample(5))
print(df.shape)
df.to_csv('data/russian-troll-tweets-master/Tweets_fr_remove_dup.csv', index=False)


# Remove stopwords for further analysis

df['content_pre'] = df['content_pre'].apply(lambda x: pl.remove_stopwords(x))


# Call get_top_n_Ngram and save the bar chart of the bi-grams and Tri-grams

Most_used_words_2 = get_top_n_Ngram(df['content_pre'], n=20, ngram_range=(2, 2))
fig = Most_used_words_2.plot(kind='bar',
                             ylabel='Count',
                             title='The 20 most frequent bi-grams in the dataset (without stopwords)').get_figure()

plt.tight_layout()

fig.savefig('images/Most_used_words_2.png')

plt.figure()
Most_used_words_3 = get_top_n_Ngram(df['content_pre'], n=20, ngram_range=(3, 3))
fig1 = Most_used_words_3.plot(kind='bar',
                              ylabel='Count',
                              title='The 20 most frequent tri-grams in the dataset (without stopwords)').get_figure()

plt.tight_layout()
fig1.savefig('images/Most_used_words_3.png')


# Lemmatize the text and make the wordcloud


Wordlem = WordNetLemmatizer()

dt = df['content_pre'].apply(Wordlem.lemmatize)
words_lem = ' '.join([word for word in dt])
create_word_cloud(words_lem, "images/Troll_face_bck_rmv.jpg", STOPWORDS, "images/WordCloud_troll.png")


# Create Wordcloud for mentions and hashtags (no need of stopwords

create_word_cloud(get_mentions_list(df['content'].to_frame()), "images/Troll_face_bck_rmv.jpg", None,
                  "images/Mentions_troll.png", col=False)
plt.figure()
create_word_cloud(get_hashtag_list(df['content'].to_frame()), "images/Troll_face_bck_rmv.jpg", None,
                  "images/Hashtags_troll.png", col=False)

# Top 20 Emoticons used and the 20 least used emoticons by the trollers and plot it as bar chart
sum_emo = plot_top_emoticon(df['content'])
