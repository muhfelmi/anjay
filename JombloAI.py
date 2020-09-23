import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import adam
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json','r',encoding='utf-8',errors = 'ignore').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding clases to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
			# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
x = list(training[:,1])
y = list(training[:,0])
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with sigmoid
model = Sequential()
model.add(Dense(128, input_shape=(len(x[1]),), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(y[0]), activation='sigmoid'))

#additional code
if (training.ndim == 1)
    training = numpy.array([training])
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fitting and saving the model
model.fit(np.array(x), np.array(y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("model created")
