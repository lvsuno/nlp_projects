import numpy as np
import pandas as pd
import pathlib
import gzip
import shutil

'''
Read russian Troll tweet data and select only tweets in english
https://github.com/fivethirtyeight/russian-troll-tweets.git

'''

'''
dir1 = pathlib.Path('data/russian-troll-tweets-master')

ite = 1
for item in dir1.iterdir():
    print(ite)
    print(item)
    df1 = pd.read_csv(item, sep=',')
    if ite == 1:
        df = pd.DataFrame(columns=df1.columns.tolist())
        print(df.columns)
    df = pd.concat([df, df1], ignore_index=True)
    ite += 1


print(f"We have processed {ite-1} files")

df.to_csv('data/russian-troll-tweets-master/IRAhandle_tweets.csv', sep=',')
'''

'''
# Read the file and extract just the english tweets
dt = pd.read_csv('data/russian-troll-tweets-master/IRAhandle_tweets.csv', sep=',')

# unique language
# print(dt['language'].unique())

# Select only tweet in English
dt_eng = dt[dt['language'] == 'English']
# save it
dt_eng.to_csv('data/russian-troll-tweets-master/tweets_eng.csv')
print(f"Data shape before dropping duplicate tweets: {dt_eng.shape}")

# Remove duplicate English tweets
dt_eng = dt_eng.drop_duplicates(subset=['content'], keep='first')
# save it
dt_eng.to_csv('data/russian-troll-tweets-master/Full_Tweets_eng_remove_dup.csv')
print(f"Data shape after dropping duplicate tweets: {dt_eng.shape}")

dt_eng_s = dt_eng[["tweet_id", "content"]]
dt_eng_s.to_csv('data/russian-troll-tweets-master/Tweets_eng_remove_dup.csv')
print(f"Data shape after selection: {dt_eng_s.shape}")


dt_fr = dt[dt['language'] == 'French']

# save it
dt_fr.to_csv('data/russian-troll-tweets-master/tweets_fr.csv')
print(f"Data shape before dropping duplicate tweets: {dt_fr.shape}")

# Remove duplicate French tweets
dt_fr = dt_fr.drop_duplicates(subset=['content'], keep='first')
# save it
dt_fr.to_csv('data/russian-troll-tweets-master/Full_Tweets_fr_remove_dup.csv')
print(f"Data shape after dropping duplicate tweets: {dt_fr.shape}")

dt_fr_s = dt_fr[["tweet_id", "content"]]
dt_fr_s.to_csv('data/russian-troll-tweets-master/Tweets_fr_remove_dup.csv')
print(f"Data shape after selection: {dt_fr_s.shape}")
'''

'''
Read a (first file) sample of January 2022 tweets. It can be found on
https://archive.org/details/archiveteam-twitter-stream-2022-01
Twitter data since 2011 can be accessed on:
https://archive.org/search?query=twitterstream&page=3&sort=-publicdate
'''
# dt = pd.read_json('data/20220101/20220101000000.json', lines=True)

# print(dt.shape)
# print(dt.columns)

'''
'created_at',  'id', 'text', 'truncated', 'in_reply_to_status_id',
       'in_reply_to_user_id', 'user', 'geo', 'coordinates', 'place', 'lang',
'''
# dt1 = dt['user'][0]['id']
# print(dt1)

# Uncompress file all 1440 files in dir 1 and put them in tmp/
'''
dir1 = pathlib.Path('data/20220101')
# pathlib.Path('data/20220101/tmp/').mkdir(parents=True, exist_ok=True)
dir2 = pathlib.Path('data/20220101/tmp/')
ite = 1
for item in dir1.iterdir():
    print(ite)
    item.stem
    if item.suffix == '.gz':
        with gzip.open(str(item), 'rb') as file:
            with open(pathlib.PurePath(dir2, item.stem), 'wb') as f_out:
                shutil.copyfileobj(file, f_out)
'''


# Read json and
dir1 = pathlib.Path('data/20220101/tmp/')
