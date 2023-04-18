import os
import pandas as pd
import pathlib
import gzip
import shutil


'''
Read russian Troll tweet data and select only tweets in english and french
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

df.to_csv('data/russian-troll-tweets-master/IRAhandle_tweets.csv', sep=',', index=False)
'''

'''
# Read the file and extract just the english tweets
dt = pd.read_csv('data/russian-troll-tweets-master/IRAhandle_tweets.csv', sep=',')
# unique language
# print(dt['language'].unique())

# Select only tweet in English
dt_eng = dt[dt['language'] == 'English']
# save it
dt_eng.to_csv('data/russian-troll-tweets-master/tweets_eng.csv', index=False)
print(f"Data shape before dropping duplicate tweets: {dt_eng.shape}")

# Remove duplicate English tweets
dt_eng = dt_eng.drop_duplicates(subset=['content'], keep='first')
# save it
dt_eng.to_csv('data/russian-troll-tweets-master/Full_Tweets_eng_remove_dup.csv', index=False)
print(f"Data shape after dropping duplicate tweets: {dt_eng.shape}")

dt_eng_s = dt_eng[["tweet_id", "content"]]
dt_eng_s.to_csv('data/russian-troll-tweets-master/Tweets_eng_remove_dup.csv', index=False)
print(f"Data shape after selection: {dt_eng_s.shape}")


dt_fr = dt[dt['language'] == 'French']

# save it
dt_fr.to_csv('data/russian-troll-tweets-master/tweets_fr.csv', index=False)
print(f"Data shape before dropping duplicate tweets: {dt_fr.shape}")

# Remove duplicate French tweets
dt_fr = dt_fr.drop_duplicates(subset=['content'], keep='first')
# save it
dt_fr.to_csv('data/russian-troll-tweets-master/Full_Tweets_fr_remove_dup.csv', index=False)
print(f"Data shape after dropping duplicate tweets: {dt_fr.shape}")

dt_fr_s = dt_fr[["tweet_id", "content"]]
dt_fr_s.to_csv('data/russian-troll-tweets-master/Tweets_fr_remove_dup.csv', index=False)
print(f"Data shape after selection: {dt_fr_s.shape}")
'''

'''
Read a (first file) sample of January 2022 tweets. It can be found on
https://archive.org/details/archiveteam-twitter-stream-2022-01
Twitter data since 2011 can be accessed on:
https://archive.org/search?query=twitterstream&page=3&sort=-publicdate
'''


# Uncompress file all files in ori and put them in dest
def uncompress_gz(ori, desti):
    """
    :param ori: pathlib data referring to the folder where the .gz files are
    :param desti: folder where to put the uncompress file
    :return:
    """
    ite = 1
    for item in ori.iterdir():
        # print(ite)
        if item.suffix == '.gz':
            with gzip.open(str(item), 'rb') as file:
                with open(pathlib.PurePath(desti, item.stem), 'wb') as f_out:
                    shutil.copyfileobj(file, f_out)


def parse_twitter(ori):
    """
    :param ori:  Folder where all the json files is
    :return: dt_en, dt_fr, twitter data split in French and English
    """
    columns_to_keep = ['created_at', 'id', 'content', 'user', 'geo', 'coordinates', 'place', 'lang']
    columns_to_keep1 = ['user_id', 'user_name', 'user_screen_name', 'user_location', 'user_description',
                        'user_created_at', 'user_geo_enabled']

    # Read json and
    ori = pathlib.Path(ori)
    ite = 1
    len_ori = len(list(ori.iterdir()))
    for item in ori.iterdir():
        if item.suffix == '.json':
            # print(ite)
            # read json file
            dt = pd.read_json(str(item), lines=True)
            # select twitter in french and english
            # dt = dt[(dt['lang'].isin(['en', 'fr']))]
            # Detect when the tweet is truncated
            dt_tr_true = dt.loc[dt.truncated == True].copy()

            # extract the extended_tweet
            df = pd.DataFrame(dt_tr_true['extended_tweet'].tolist())
            # Put the extended_tweet in the field 'content'
            dt_tr_true.index = df.index
            dt_tr_true.loc[:, 'content'] = df.loc[:, 'full_text']
            # For the non-truncated tweet, put the text in the field content

            dt_tr_false = dt.loc[dt.truncated == False].copy()
            # dt_tr_false.index = dt_tr_true.index
            dt_tr_false.loc[:, 'content'] = dt_tr_false.loc[:, 'text']
            dt = pd.concat([dt_tr_false, dt_tr_true], ignore_index=True)
            del dt_tr_false, dt_tr_true
            dt = dt[columns_to_keep]
            df = pd.DataFrame(dt['user'].tolist())
            df = df.add_prefix('user_')
            # print(df.columns)
            df = df[columns_to_keep1]
            dt.drop('user', inplace=True, axis=1)
            df = dt.join(df)
            del dt
            df = df.drop_duplicates(subset=['content'], keep='first')
            # print(data.columns)

            if ite == 1:
                data = pd.DataFrame(columns=df.columns.tolist())
            df['user_geo_enabled'] = df['user_geo_enabled'].astype("boolean")
            data = pd.concat([data, df], ignore_index=True)
            print(f"######### {ite} files treated on {len_ori} #####")
            ite += 1
    # dt_en = data[data['lang'] == 'en']
    # dt_fr = data[data['lang'] == 'fr']
    # del data

    return data  # dt_en, dt_fr


'''
dir1 = pathlib.Path('data/20220225')
pathlib.Path('data/20220225/tmp/').mkdir(parents=True, exist_ok=True)
dir2 = pathlib.Path('data/20220225/tmp/')

uncompress_gz(dir1, dir2)
'''

# data_en, data_fr = parse_twitter('data/20220225/tmp/')

### 24
'''
dir1 = pathlib.Path('data/20220224')
pathlib.Path('data/20220224/tmp/').mkdir(parents=True, exist_ok=True)
dir2 = pathlib.Path('data/20220224/tmp/')

# uncompress_gz(dir1, dir2)
data = parse_twitter('data/20220224/tmp/')
data.to_json('data/20220224/20220224_full.json')
'''
### 25
'''
dir1 = pathlib.Path('data/20220225')
pathlib.Path('data/20220225/tmp/').mkdir(parents=True, exist_ok=True)
dir2 = pathlib.Path('data/20220225/tmp/')

uncompress_gz(dir1, dir2)
data = parse_twitter('data/20220225/tmp/')
data.to_json('data/20220225/20220225_full.json')
'''

DATA_URL = ('data/20220224/20220224_full.json')
data = pd.read_json(DATA_URL)
print(data.shape)

