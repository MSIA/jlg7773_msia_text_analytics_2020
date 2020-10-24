import numpy as np
import pandas as pd
import json
import re
import string
import fasttext
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize



def preprocess_fasttext():

	# organize the input text in fasttext required format, which is:__label__1 text
	reviews = open('yelp_dataset/yelp_academic_dataset_review.json', encoding="utf8").readlines()[:500000]
	fasttext_entry_list =[]
	for review in reviews:
		star, text_cleaned = process_one_review_fasttext(review)
		entry_fasttext = '__label__' + str(star) + ' ' + text_cleaned
		fasttext_entry_list.append(entry_fasttext)

	# do a train test split
	fasttext_train = fasttext_entry_list[0:400000]
	len(fasttext_train)
	fasttext_test = fasttext_entry_list[400001:500000]
	len(fasttext_test)

	with open('./fasttest_data/fasttext_train.txt', 'w') as f_train:
		for item in fasttext_train:
			f_train.write("%s\n" % item)

	with open('./fasttest_data/fasttext_test.txt', 'w') as f_test:
		for item in fasttext_test:
			f_test.write("%s\n" % item)



def process_one_review_fasttext(review):
	"""
	process each review, extract lable and text
	:param texts: input list of texts, each is one review
	:return:
	"""
	review_dict = json.loads(review)
	# (2) get number of labels
	star = review_dict['stars']
	text = review_dict['text']
	#
	return star, text



if __name__ == "__main__":

	preprocess_fasttext()
	# I experimented with following hyper-parameters
	# lr
	# epoch (default is 5)
	# wordNgrams

	# Experiment 1:
	model = fasttext.train_supervised(input='./fasttest_data/fasttext_train.txt',lr=1.0,epoch=5)

	# get model performance
	result = model.test('./fasttest_data/fasttext_test.txt')
	precision = result[1]
	recall = result[2]
	print("F1 score: %0.4f" % (2 * precision * recall / (precision + recall)))
	# f1-score is 0.6355

	# Experiment 2:
	model = fasttext.train_supervised(input='./fasttest_data/fasttext_train.txt', lr=1.0, epoch=25)

	# get model performance
	result = model.test('./fasttest_data/fasttext_test.txt')
	precision = result[1]
	recall = result[2]
	print("F1 score: %0.4f" % (2 * precision * recall / (precision + recall)))
	# F1 score is 0.6102, model starts to overfit

	# Experiment 3:
	model = fasttext.train_supervised(input='./fasttest_data/fasttext_train.txt', lr=1.0, epoch=5, wordNgrams=2)

	# get model performance
	result = model.test('./fasttest_data/fasttext_test.txt')
	precision = result[1]
	recall = result[2]
	print("F1 score: %0.4f" % (2 * precision * recall / (precision + recall)))
	# F1 score is 0.6300

	# model = fasttext.train_supervised(input='./fasttext_train.txt',lr=1.0,epoch=25)
	#
	# # get model performance
	# result = model.test('./fasttext_test.txt')
	# precision = result[1]
	# recall = result[2]
	# print("F1 score: %0.4f" % (2 * precision * recall / (precision + recall)))



