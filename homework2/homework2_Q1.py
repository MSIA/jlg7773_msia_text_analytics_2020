import os
import glob
import re
import string
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec


def preprocess(file_list):

	"""
	tokenize, normalize and remove number from the original text corpus
	format the output as [[],[]...], needed for training the word2vec model
	:param file_list:
	:return:
	"""
	file_list_formatted = []
	for i in file_list:
		# convert to lower cases
		i = i.lower()
		# remove numbers
		re.sub(r'\d+', '', i)
		# remove punctuation
		i = re.sub('[' + string.punctuation + ']', '', i)
		# remove white space
		i.strip()
		# tokenize
		tokens = word_tokenize(i)
		# append to list
		file_list_formatted.append(tokens)

	return file_list_formatted


def train_word2vec(file_list_formatted,size,model_name,model_file_name):

	"""
	:param file_list_formatted: formatted list of training text corpus, [[],[],..]
	:return: a trained model object
	"""
	if model_name == 'cbow':
		sg = 0
	elif model_name == 'skipgram':
		sg = 1
	model = Word2Vec(file_list_formatted, size=size, sg=sg, window=5, min_count=1, workers=12)
	model.save(model_file_name)

	return model


def get_3_closest_words(input_word_lists, model_object):

	"""
	This functions takes a trained model object and a list of words and outputs
	the top 3 closest neighboring words
	:param input_word_lists: a list of words that we want to get the similar words for
	:param model_object: trained model object
	:return:
	"""
	results_df = pd.DataFrame()

	for i, word in enumerate(input_word_lists):

		result = model_object.most_similar(word)[:3]

		results_df[word] = result

	results_df.index = ['1st closest word', '2nd closest word', '3rd closest word']

	print(results_df)

	return results_df


if __name__ == "__main__":

	files_list = []
	# root = os.getcwd() + '/homework1/20news-bydate/20news-bydate-test/talk.religion.misc/'
	root = os.getcwd() + '/homework1/20news-bydate/20news-bydate-train/'
	for path, subdirs, files in os.walk(root):
		for name in files:
			files_list.append(os.path.join(path, name))

	files_text_list = []
	for i in files_list:
		with open(i, 'r', encoding="utf8", errors="ignore") as f:
			file = f.read()
			files_text_list.append(file)


	file_list_formatted = preprocess(files_text_list)

	# train 3 models with different parameters
	model_0 = train_word2vec(file_list_formatted,50,'cbow','model1_cbow_100')
	model_1 = train_word2vec(file_list_formatted,100,'cbow','model1_cbow_100')
	model_2 = train_word2vec(file_list_formatted, 500, 'cbow', 'model1_cbow_500')
	model_3 = train_word2vec(file_list_formatted, 100, 'skipgram', 'model1_skipgram_100')
	model_list = [model_0,model_1,model_2,model_3]
	print("finished training 4 models")

	total_results = pd.DataFrame(index=['1st closest word', '2nd closest word', '3rd closest word'])
	for model in model_list:
		##
		input_word_lists = ['good','bad','music','home','road','red','country','one','two','children']
		##
		results_df = get_3_closest_words(input_word_lists, model)
		##
		total_results=total_results.append(results_df)
	#
	total_results.to_csv('./homework2_q1_eval_results.csv')
	print(total_results)




