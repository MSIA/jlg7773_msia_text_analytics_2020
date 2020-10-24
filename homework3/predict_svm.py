import numpy as np
import pandas as pd
import re
import string
import pickle
import json
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


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

def predict_svm(X_input):

	# load the tf-idf pipe object
	pipe = pickle.load(open('./tfidf_pipe.pkl', 'rb'))
	# transform input corpus into tf_idf representation
	X_input_tfidf = pipe.transform([X_input])

	# load the best-performing model object
	tmo = pickle.load(open('./best_model_logistic.pkl', 'rb'))

	# predict the label using the transformmed input
	predicted_label = tmo.predict(X_input_tfidf)
	predicted_prob = tmo.predict_proba(X_input_tfidf)

	# write the output in json format
	output = dict()
	output['label'] = predicted_label[0]
	output['confidence_score'] = list(predicted_prob[0])


	print("input tokenized texts:")
	print(X_input)
	print("predicted label:")
	print(predicted_label)
	print("predicted probability:")
	print(predicted_prob)

	# return the output dictionary
	return output


if __name__ == "__main__":

	with open('prediction_output.json','a') as file:

		text_input = "this movie is just so-so"
		cleaned_text = clean_and_tokenize(text_input)
		# input should be text (it will be transformed into tf-idf vector)
		output = predict_svm(str(cleaned_text))
		json.dump(output,file)

		text_input = "this movie is fantastic"
		cleaned_text = clean_and_tokenize(text_input)
		# input should be text (it will be transformed into tf-idf vector)
		output = predict_svm(str(cleaned_text))
		json.dump(output,file)

		text_input = "I don't like this restaurant"
		cleaned_text = clean_and_tokenize(text_input)
		# input should be text (it will be transformed into tf-idf vector)
		output = predict_svm(str(cleaned_text))
		json.dump(output,file)

		text_input = "This is our to-go place"
		cleaned_text = clean_and_tokenize(text_input)
		# input should be text (it will be transformed into tf-idf vector)
		output = predict_svm(str(cleaned_text))
		json.dump(output,file)

		text_input = "I am a big fan of this place"
		cleaned_text = clean_and_tokenize(text_input)
		# input should be text (it will be transformed into tf-idf vector)
		output = predict_svm(str(cleaned_text))
		json.dump(output,file)
