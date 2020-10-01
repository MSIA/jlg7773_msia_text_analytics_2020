import os,time
import glob
import nltk
import spacy
from spacy.lang.en import English
import stanza
import multiprocessing



# Q1 asks us to compare 3 NLP libraries on the tasks of tokenization, stemming and POS tagging
# compare run times, ease of parallelization and performance review

def tokenization(library_name, file):

	"""
	:param library_name: The name of library used to perform the task
	:param file_list: the raw text used to perform the task
	:return: time it takes to finish the task for the same text corpus
	"""

	# setup each library for English tokenization
	nlp_spacy = English()
	nlp_stanza = stanza.Pipeline('en')

	# perform tokenization using library of choice
	start_time = time.time()

	if library_name == 'nltk':
		tokens = nltk.word_tokenize(file)
	elif library_name == 'spacy':
		tokens = nlp_spacy(file)
	elif library_name == 'stanza':
		tokens = nlp_stanza(file)

	finish_time=time.time()
	run_time = finish_time-start_time
	return run_time


def stemming(library_name, file):

	"""
	:param library_name: name of the library used to perform the task
	:param file_list: the raw text used to perform the task
	:return: time it takes to run the stemming job
	"""
	tokens = nltk.word_tokenize(file)

	if library_name == 'nltk':
		ps = nltk.PorterStemmer()
		stemmed_words=[]
		for word in tokens:
			stemmed_words.append(ps.stem(word))

	return stemmed_words


def pos_tagging(library_name, file):

	"""
	:param library_name: library name chosen to perform the task
	:param file_list: raw text used to perform the task
	:return: run time it takes to finish the task
	"""

	start_time = time.time()

	if library_name == 'nltk':
		tokens = nltk.word_tokenize(file)
		pos_tags = nltk.pos_tag(tokens)

	if library_name == 'spacy':
		nlp = spacy.load('en_core_web_sm')
		tokens = nlp(file)
		pos_tags=[]
		for token in tokens:
			pos_tags.append((token.text, token.pos_))

	if library_name == 'stanza':
		nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
		doc = nlp(file)
		pos_tags=[]
		for sent in doc.sentences:
			for word in sent.words:
				pos_tags.append((word.text, word.upos))

	end_time = time.time()
	run_time = end_time - start_time

	return run_time

def parallelization_nltk(file):

	pos_tags = pos_tagging(library_name='nltk',file=file)

	return pos_tags

def parallelization_spacy(file_list):

	nlp = spacy.load('en_core_web_sm')
	for doc in nlp.pipe(file_list, n_threads=16, batch_size=30):
		pos_tags = [[token.text, token.pos_] for token in doc]

	return pos_tags


if __name__ == "__main__":

	# prepare raw text used for following task:
	path = os.getcwd() + '/20news-bydate/20news-bydate-test/talk.religion.misc/*'
	files = glob.glob(path)

	file_list=[]
	for i in files:
		with open(i, 'r', encoding="utf8", errors="ignore") as f:
			file = f.read()
			file_list.append(file)
	#
	##############################
	# Task 1: compare tokenization
	##############################
	file = file_list[0]
	time_nltk = tokenization(library_name='nltk',file=file)
	time_spacy = tokenization(library_name='spacy',file=file)
	time_stanza = tokenization(library_name='stanza',file=file)
	#
	print("Tokenization task run time for nltk, spacy and stanze, on the same one file")
	print(time_nltk,time_spacy,time_stanza)

	# ##############################
	# # Task 2: compare POS tagging
	# ##############################
	file = file_list[0]
	time_nltk = pos_tagging(library_name='nltk',file=file)
	time_spacy = pos_tagging(library_name='spacy',file=file)
	time_stanza = pos_tagging(library_name='stanza',file=file)

	print("POS tagging task run time for nltk, spacy and stanza:")
	print(time_nltk,time_spacy,time_stanza)

	############################
	#Task 3: Parallelization
	############################
	count = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(count)

	start_time = time.time()
	result_nltk = pool.map(parallelization_nltk,file_list)
	finish_time= time.time()
	run_time_nltk = finish_time - start_time

	start_time = time.time()
	result_spacy = parallelization_spacy(file_list)
	finish_time = time.time()
	run_time_spacy = finish_time - start_time

	print("Parallelization run time for nltk, spacy")
	print(run_time_nltk, run_time_spacy)

	##################################
	# Task 3 (comparison): without parallelization
	##################################

	start_nltk =time.time()
	for file in file_list:
		results = parallelization_nltk(file=file)
	finish_nltk = time.time()
	runtime_nltk = finish_nltk - start_nltk

	start_spacy = time.time()
	for file in file_list:
		results = pos_tagging(library_name='spacy', file=file)
	finish_spacy = time.time()
	runtime_spacy = finish_spacy - start_spacy

	print("No Parallelization run time for nltk, spacy")
	print(runtime_nltk, runtime_spacy)






