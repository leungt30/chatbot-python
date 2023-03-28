#Followed a tutorial made by tech with tim 
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x  = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds) #pattern and its corresponding tag stored in docs_x and docs_y
            docs_y.append(intent["tag"])

        if intent ["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    #the training cant handle strings, only numbers so we make an array that holds whether a particular word exist or not in the given string
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1) #1 in bag if it exists
            else:
                bag.append(0) #0 in bag if it doesnt

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    #change to arrays
    training = numpy.array(training)
    output = numpy.array(output)

    #save so we dont have to re train every time
    with open("data.pickle", "wb") as f:
        pickle.dump((words,labels,training,output),f)


#model 
tensorflow.compat.v1.reset_default_graph()

#input, 3 hidden layers, output
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]),activation="softmax")
net = tflearn.regression(net)

#training model 
model = tflearn.DNN(net)

#comment this stuff when updating intents

model.load("model.tflearn") #load model if no need to update intents
    # pass model the training data, uncomment to retrain
# model.fit(training, output, n_epoch=1000,batch_size=8,show_metric=True)
# model.save("model.tflearn")



def bag_of_words(s,words): #generate bag of words
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chat(inp):
    # print("Start talking with the bot! (type quit to stop)")
    # while True:
        # inp = input("You: ")
    results = model.predict([bag_of_words(inp,words)])[0] #results is a list of probabilities
    results_index = numpy.argmax(results) #gives index of largest probability
    inputTag = labels[results_index]
    
    #confidence check, if not too confident just say you dont understand
    if results[results_index] > 0.7:
        #find corresponding responses
        for tag in data["intents"]:
            if tag['tag'] == inputTag:
                responses = tag['responses']
        #print one of the responses
        # print(random.choice(responses))
        return (random.choice(responses))
    else:
        # print("I don't quite understand. Try rephrasing it or try a new question")
        return ("I don't quite understand. Try rephrasing it or try a new question")

print("Hey, I am a chatbot made by Timothy Leung. You can ask me questions and I'll answer as if I am Tim. Sometimes my responses change so ask me a question multiple times to learn more about me")
while (True):
    inp = input("You: ")
    if inp == "quit":
        break
    print(chat(inp))