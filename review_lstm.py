import numpy as np 
import tflearn 
import pickle 
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.core import input_data, dropout, fully_connected
from sklearn.metrics import classification_report


#globals
len_lexicon = 11841

#loading dataset from review.pickle
print('loading dataset from pickle file')
with open('review.pickle', 'r') as f:
	dataset = pickle.load(f)

#padding each sequence to length 10
x_store = dataset[:, 0]
x_store = pad_sequences(x_store, maxlen = 100, value = 0)
y_store = dataset[:, 1]

x = np.array([])
y = np.array([])

for element in x_store:
	x = np.append(x, np.array(element))

for element in y_store:
	y = np.append(y, np.array(element))


x = x.reshape([-1, 100])
y = y.reshape([-1, 2])

#70% training data with 10% validation data and 30% test data
test_size = 0.3
test_length = int(test_size * len(dataset))

# print('dataset length', len(dataset))
# print('test length', test_length)

train_data = x[:-test_length]
train_label = y[:-test_length]
test_data = x[-test_length:]
test_label = y[-test_length:]

# print(test_data)
# print(test_label)

#defining architecture of LSTM Network
input_data = tflearn.input_data(shape = [None, 100], name = 'input')
embed_layer = embedding(input_data, input_dim = len_lexicon, output_dim = 100)
lstm_layer_1 = tflearn.lstm(embed_layer, 512, return_seq = True)
dropout_layer_1 = tflearn.dropout(lstm_layer_1, 0.5)
lstm_layer_2 = tflearn.lstm(dropout_layer_1, 512, return_seq = True)
dropout_layer_2 = tflearn.dropout(lstm_layer_2, 0.5)
lstm_layer_3 = tflearn.lstm(dropout_layer_2, 512)
dropout_layer_3 = tflearn.dropout(lstm_layer_3, 0.5)
fc = fully_connected(dropout_layer_3, 2, activation = 'softmax')
lstm_net = tflearn.regression(fc, optimizer = 'adam', loss = 'categorical_crossentropy')

#training
model = tflearn.DNN(lstm_net, clip_gradients = 0.5, tensorboard_verbose = 2)
model.fit(train_data, train_label, validation_set = (test_data, test_label), show_metric = True, batch_size = 128, n_epoch = 5)
# print('evaluation metric on test data : ')
# print(model.evaluate(test_data, test_label))
predY = np.array(model.predict(test_data))
pY = np.array([])
tY = np.array([])
for res in predY:
	pY = np.append(pY, np.argmax(res))
for res in test_label:
	tY = np.append(tY, np.argmax(res))
print(classification_report(tY, pY))
print('accuracy : ', model.evaluate(test_data, test_label))