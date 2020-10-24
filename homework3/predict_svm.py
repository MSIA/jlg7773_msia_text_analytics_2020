import numpy as np
import pandas as pd
import re
import string
import pickle
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


def predict_svm(X_input):

	# load the tf-idf pipe object
	pipe = pickle.load(open('./tfidf_pipe.pkl', 'rb'))
	# transform input corpus into tf_idf representation
	X_input_tfidf = pipe.transform([X_input])

	# load the best-performing model object
	tmo = pickle.load(open('./best_model.pkl', 'rb'))

	# predict the label using the transformmed input
	predicted_label = tmo.predict(X_input_tfidf)

	print("input tokenized texts:")
	print(X_input)
	print("predicted label:")
	print(predicted_label)


if __name__ == "__main__":

	data_df = pd.read_csv('./cleaned_100.csv').drop(columns=['Unnamed: 0'], axis=1)
	X_input = data_df['text'][0]
	# input should be text (it will be transformed into tf-idf vector)
	predict_svm(X_input)