import pandas as pd
import string
import os
import numpy as np
import tweepy
import json
from tqdm import tqdm
import random
import re

DATA_DIR = '/home/ubuntu/CS224u_Final_Project/data'
output_path = os.path.join(DATA_DIR, 'clean_unlabeled_tweets.csv')
input_path = os.path.join(DATA_DIR,'corona_tweets_31_hydrated-002.json')

# read data 
indices = list(np.arange(400000, 500000))
print("Reading json file")
with open(input_path, encoding='utf-8') as json_file:
	data = json_file.readlines()
	data = [data[i] for i in indices]
	print(len(data))
	data = list(map(json.loads, data))
print("reading complete")

# Loading in the hydrated tweets into a dataframe
tweets_df = pd.DataFrame(data)

# Creating a dictionary for thematic replacement
replace_dict = pd.read_csv(os.path.join(DATA_DIR, 'hs_AsianPrejudice_hashtagsThematicReplacements.csv'))
dict_replace = replace_dict.set_index('Hashtag')['Replacement'].to_dict()

# Create new column for clean text
tweets_df['processed_txt'] = np.nan

# function to remove url and user name, replace hashtags, lowercase the tweet
def clean_tweet(line):
	line = line.lower()
	# replace most common hashtags
	tokens = [dict_replace.get(word.strip(string.punctuation.replace('#', '')).lower(), word) for word in line.split()]
	# remove all other hashtags and user names
	for token in tokens:
		if token[0] == '#':
			clean_tokens.append('#HASHTAG')
		else:
			clean_tokens.append(token)
	# collapse into a string and lower case
	clean_text = " ".join(clean_tokens)
	return clean_text

# For re-tweets, full_text does not contain the entire tweet, so we have to go to the actual tweet to get the full$
for index, row in tqdm(tweets_df.iterrows()):
	line = None
	tokens, clean_tokens = [], []
	if tweets_df['retweeted_status'].isnull()[index]:
		line = tweets_df['full_text'][index]
	else:
		line = row['retweeted_status']['full_text']
	tweets_df.loc[index, 'processed_txt'] = clean_tweet(line)
	if (index+1) % 10 == 0:
		print("processed {} tweets".format(index+1))

columns_to_select = ['processed_txt', 'id_str']
tweets_df.to_csv(output_path, header=False, columns=columns_to_select, mode='a', index=False)

processed = pd.read_csv(output_path)
print("total tweets processed {}".format(processed.shape))
