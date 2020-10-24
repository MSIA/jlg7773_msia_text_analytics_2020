import numpy as np
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


def train_and_eval(ngram_range_low=1, ngram_range_hi=1):

	data_df = pd.read_csv('./cleaned_100.csv').drop(columns=['Unnamed: 0'], axis=1)

	X_train, X_test, y_train, y_test = train_test_split(data_df['text'], data_df['star'], test_size = 0.2, random_state=42)

	pipe = Pipeline([('count', CountVectorizer(ngram_range=(ngram_range_low,ngram_range_hi))),
					 ('tfid', TfidfTransformer())]).fit(X_train)
	# transform input corpus into tf_idf representation
	X_train_tfidf = pipe.transform(X_train)

	svm_c_list = [1,5]
	for svm_c in svm_c_list:
		# fit logistic regression model
		clf = LinearSVC(C=svm_c, random_state=1).fit(X_train_tfidf, y_train)

		# evaluate model performance on test set
		X_test_tfidf = pipe.transform(X_test)
		y_pred = clf.predict(X_test_tfidf)

		# get performance measures
		cm = confusion_matrix(y_test,y_pred,labels=[1,2,3,4,5])
		# precision, recall, f1 is represented by a numpy array for 5 classes
		precision = np.diag(cm)/np.sum(cm,axis=0)
		recall = np.diag(cm)/np.sum(cm,axis=1)
		f1 = 2 * np.multiply(precision,recall)/(precision+recall)
		# aggregate TP for all classes and compute micro_f1
		micro_f1 = np.sum(np.diag(cm))/np.sum(np.sum(cm,axis=0))

		print("precision, recall, f1 for all 5 classes:")
		print(precision)
		print(recall)
		print(f1)
		print("micro_f1 is:")
		print(micro_f1)

		with open('performance_results.txt', 'a') as file:
			file.write("Linear SVM Model:\n")
			file.write("Model parameters c:\n")
			file.write(str(svm_c)+'\n')
			file.write("ngram parameter is:\n")
			file.write(str(ngram_range_low)+str(ngram_range_hi)+'\n')
			file.write("precision for 5 classes is:\n")
			file.write(np.array2string(precision)+'\n')
			file.write("recall for 5 classes is:\n")
			file.write(np.array2string(recall)+'\n')
			file.write("f1 for 5 classes is:\n")
			file.write(np.array2string(f1)+'\n')
			file.write("micro-f1 is:\n")
			file.write(np.array2string(micro_f1)+'\n')


def convertTuple(tup):
	str =  ''.join(tup)
	return str


if __name__ == "__main__":

	train_and_eval(ngram_range_low=1, ngram_range_hi=1)
	train_and_eval(ngram_range_low=1, ngram_range_hi=2)