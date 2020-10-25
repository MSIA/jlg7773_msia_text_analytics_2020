import os
import json
import matplotlib.pyplot as plt
from statistics import mean
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

	return star, text


def get_stats():

	"""
	based on the file "yelp_academic_dataset_review.json", extract following info:
	(1) number of documents
	(2) number of labels
	(3) label distribution
	(4) average word length of documents
	:return:
	"""

	reviews = open('yelp_dataset/yelp_academic_dataset_review.json', encoding="utf8").readlines()[:500000]
	star_list = []
	text_list =[]
	doc_len_list=[]
	for review in reviews:
		star, text = process_one_review(review)
		star_list.append(star)
		text_list.append(text)
		doc_len = len(word_tokenize(text))
		doc_len_list.append(doc_len)


	# (1) get total number of documents
	num_docs = len(reviews)
	# (2) get number of labels
	num_labels = len(set(star_list))
	# (3) get label distribution
	star_cnt_list =[]
	for i in list(set(star_list)):
		star_cnt_list.append(star_list.count(i))
	print("count for each star:")
	print(star_cnt_list)
	# plot the distribution
	plt.bar(list(set(star_list)),star_cnt_list,label='lable (stars) distribution')
	plt.savefig('./label_distribution.png')
	# (4) average word length of documents
	average_doc_len = mean(doc_len_list)
	print("number of documents is %d" % num_docs)
	print("number of labels is %d" % num_labels)
	print("distinct label values are:")
	print(set(star_list))
	print("average review length is %d" % average_doc_len)

if __name__=="__main__":
	get_stats()

