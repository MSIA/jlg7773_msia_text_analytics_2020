import os
import glob
import re


def find_email(text):

	lst = re.findall('\S+@\S+',text)
	return lst


def find_dates(text):
	"""
	this function tries to find all possible dates from the text
	:param text: input raw text
	:return: a list of matched date text
	"""
	# Notes: The first regex is to match dates like 01/02/2020,
	results = re.findall(r'(\d+/\d+/\d+|\S+ \d{1,2}\S{2,2} \d{2,4}|\d{1,2}\s[JFMASOND]\w{2}\s\d{4})|\d{4}[JFMASOND]\w{2}\d{2}|[JFMASOND]\w+ \d{4}',text)
	# p = re.findall(r'\d{4}[JFMASOND]\w{2}\d{2}',text)
	# p = re.findall(r'\d{4}[JFMASOND]\w{2}\d{2}', text)
	# p = re.findall(r'[JFMASOND]\w+ \d{4}', text)

	return results



if __name__ == "__main__":

	# prepare raw text used for following task:
	path = os.getcwd() + '/20news-bydate/20news-bydate-test/talk.politics.mideast/*'
	files = glob.glob(path)

	file_list=[]
	for i in files:
		with open(i, 'r', encoding="utf8", errors="ignore") as f:
			file = f.read()
			file_list.append(file)

	############
	# find emails
	############
	email_results = []
	for file in file_list:
		results = find_email(file)
		email_results = email_results + results
	# ############
	# # find dates
	# ############
	date_results = []
	for file in file_list:
		results = find_dates(file)
		date_results = date_results + results



