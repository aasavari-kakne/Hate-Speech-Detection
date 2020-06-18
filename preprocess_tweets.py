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
input_path = os.path.join(DATA_DIR, 'corona_tweets_31_hydrated-002.json')
replace_file = os.path.join(DATA_DIR, 'hs_AsianPrejudice_hashtagsThematicReplacements.csv')

# Creating a dictionary for thematic replacement
replace_dict = pd.read_csv(replace_file)
dict_replace = replace_dict.set_index('Hashtag')['Replacement'].to_dict()
# function to remove url and user name, replace hashtags, lowercase the tweet
def clean_tweet(line):
	line = line.lower()
	# replace most common hashtags
	tokens = [dict_replace.get(word.strip(string.punctuation.replace('#', '')).lower(), word) for word in line.split()]
	# remove all other hashtags and user names
	for token in tokens:
		if token[0] == '#' and token[1:].islower():
			clean_tokens.append('#HASHTAG')
		elif token[0] == '#' and token[1:].isupper():
			clean_tokens.append(token[1:])
		else:
			clean_tokens.append(token)
	return " ".join(clean_tokens)

start_index = 0
end_index  = 1400000
batch_size  = 100000

while(start_index < end_index):
	# step 1 : read data
	indices = list(np.arange(start_index, start_index+batch_size))
	with open(input_path, encoding='utf-8') as json_file:
		data = json_file.readlines()
		data = [data[i] for i in indices]
		print("Processing batch of size {}".format(len(data)))
		data = list(map(json.loads, data))
	tweets_df = pd.DataFrame(data)
	tweets_df['processed_txt'] = np.nan
	# step 2 : process all the tweets in the batch
	for index, row in tqdm(tweets_df.iterrows()):
		line = None
		tokens, clean_tokens = [], []
		if tweets_df['retweeted_status'].isnull()[index]:
			line = tweets_df['full_text'][index]
		else:
			line = row['retweeted_status']['full_text']
		tweets_df.loc[index, 'processed_txt'] = clean_tweet(line)
		if (index+1) % 100 == 0:
				print("in current batch, processed {} tweets".format(index+1))
	# step 3 : write batch to file
	columns_to_select = ['processed_txt', 'id_str']
	tweets_df.to_csv(output_path, header=False, columns=columns_to_select, mode='a', index=False)
	# step 4 : check if it is written properly
	processed = pd.read_csv(output_path)
	print("total tweets processed till now {}".format(processed.shape))
	# step 5 : increment start_index
	start_index += batch_size
