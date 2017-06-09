import os
import numpy as np 
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import random 
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

#globals
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
dataset = []

#check if dataset is present 
print('checking dataset')
folders = []
for folder in os.listdir(os.getcwd()):
	if os.path.isdir(folder):
		folders.append(folder)
if 'op_spam_v1.4' in folders:
	print('dataset found')
else:
	print('no valid dataset found')

lexicon = []

#fetch and group data for true reviews
print('fetching positive reviews')
true_locations = ['positive_polarity/truthful_from_TripAdvisor', 'negative_polarity/truthful_from_Web']
true_text = []
for loc in true_locations:	
	true_dirs = 'op_spam_v1.4/' + loc
	for true_folder in os.listdir(true_dirs):
		# print('inside folder : ', tp_folder)
		parent = true_dirs + '/' + true_folder
		true_store = os.listdir(parent)
		for text_file in true_store:
			file_name = parent + '/' + text_file
			with open(file_name, 'r') as f:
				contents = f.readlines()
				for line in contents:
					try:
						words = word_tokenize(line.lower())
					except Exception as e:
						pass
					for word in words :
						if word not in stop_words:
							true_text.append(word)
true_text = [lemmatizer.lemmatize(word) for word in true_text]

#fetch and group data for false reviews
print('fetching negative reviews')
false_locations = ['positive_polarity/deceptive_from_MTurk', 'negative_polarity/deceptive_from_MTurk']
false_text = []
for loc in false_locations:	
	false_dirs = 'op_spam_v1.4/' + loc
	for false_folder in os.listdir(false_dirs):
		# print('inside folder : ', tp_folder)
		parent = false_dirs + '/' + false_folder
		false_store = os.listdir(parent)
		for text_file in false_store:
			file_name = parent + '/' + text_file
			with open(file_name, 'r') as f:
				contents = f.readlines()
				for line in contents:
					try:
						words = word_tokenize(line.lower())
					except Exception as e:
						pass
					for word in words :
						if word not in stop_words:
							false_text.append(word)
false_text = [lemmatizer.lemmatize(word) for word in false_text]

#preparing lexicon
print('preparing lexicon')
for word in true_text:
	if word not in lexicon:
		lexicon.append(word)
for word in false_text:
	if word not in lexicon:
		lexicon.append(word)

print('length of lexicon : ', len(lexicon))

#converting the true and false review datasets into index encodings
print('converting true list')
true_list = []

for word in true_text:
	if word in lexicon:
		idx_word = lexicon.index(word)
		true_list.append(idx_word)
for i in range(0, len(true_list), 100):
	dataset.append([true_list[i:i+100], [1, 0]])

print('converting false list')
false_list = []

for word in false_text:
	if word in lexicon:
		idx_word = lexicon.index(word)
		false_list.append(idx_word)
for i in range(0, len(false_list), 100):
	dataset.append([false_list[i:i+100], [0, 1]])

#shuffling dataset
dataset = np.random.permutation(dataset)

#writing dataset to review.pickle
print('writing dataset to pickle file')
with open('review.pickle', 'w') as f:
	pickle.dump(dataset, f)

print('prep_data complete')
