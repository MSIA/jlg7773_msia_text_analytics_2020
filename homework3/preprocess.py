import os
import json
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from statistics import mean
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize

def process_one_review(review):
	"""
	process each review, extract lable and text
	:param texts: input list of texts, each is one review
	:return:
	"""
	review_dict = json.loads(review)
	# (2) get number of labels
	star = review_dict['stars']
	text = review_dict['text']
	text_cleaned = clean_and_tokenize(text)
	#
	return star, text_cleaned


def clean_and_tokenize(text):
	"""
	tokenize, normalize and remove number from the original text corpus
	format the output as [[],[]...], needed for training the word2vec model
	:param file_list:
	:return:
	"""
	# convert to lower cases
	i = text.lower()
	# remove numbers
	re.sub(r'\d+', '', i)
	# remove punctuation
	i = re.sub('[' + string.punctuation + ']', '', i)
	# remove white space
	i.strip()
	# tokenize
	tokens = word_tokenize(i)
	# append to list
	return tokens


def preprocess():
	reviews = open('yelp_dataset/yelp_academic_dataset_review.json', encoding="utf8").readlines()[:100]
	star_list = []
	text_list = []
	for review in reviews:
		star, text_cleaned = process_one_review(review)
		star_list.append(star)
		text_list.append(text_cleaned)

	# save to a dataframe and then to csv
	cleaned = pd.DataFrame(list(zip(text_list,star_list)), columns=['text','star'])
	cleaned.to_csv('./cleaned_100.csv')


if __name__ == "__main__":
	preprocess()



