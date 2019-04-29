# imports

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import os
# things needed from Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import argparse
from multiprocessing import Queue
import queue
from queue import PriorityQueue

tf.logging.set_verbosity(tf.logging.ERROR)

# initiate parser
parser = argparse.ArgumentParser()

# add long and short argument
parser.add_argument("--question", "-q", help="question for chatbot")
# read arguments from the command line
args = parser.parse_args()

#userContext = {"lastintent": "none", "namedentity": "service", "title": "SDE2", "nextaction":"none"}
class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
 
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

class ResolvedContext:
    def __init__(self, json_def):
        s = json.loads(json_def)
        self.lastintent = None if 'lastintent' not in s else s['lastintent']
        self.namedentity = None if 'namedentity' not in s else s['namedentity']
        self.title = None if 'title' not in s else s['title']
        self.nextaction = None if 'nextaction' not in s else s['nextaction']

class UserContext:
    def __init__(self,lastintent,namedentity,title,nextaction):
        self.lastintent = lastintent
        self.namedentity = namedentity
        self.title = title
        self.nextaction = nextaction
        
    def __str__(self):
        return self.lastintent

# restore all saved data structures
import pickle
data = pickle.load(open('training_data', 'rb'))

lastintent = "outages"
files = []
resolvedContexts = []
processedFiles = None

import os
if os.listdir('c:\\MachineLearning_Projects\\processed\\'):
    processedFiles = [".".join(f.split(".")[:-1]) for f in os.listdir('c:\\MachineLearning_Projects\\processed\\') if os.path.isfile(f)]

#processedFiles = filter(lambda x: x.endswith('.txt'), os.listdir('c:\\MachineLearning_Projects\\processed\\'))
userContext = UserContext("none",  "service", "SDE2", "none")

def serialize_json(instance=None, path=None):
    dt = {}
    dt.update(vars(instance))

    with open(path, "w") as file:
        json.dump(dt, file)


if not os.path.isfile("c:\MachineLearning_Projects\processed.txt"):
    serialize_json(userContext, r"c:\MachineLearning_Projects\test.json")

words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file

with open('intents.json') as json_data:
    intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


def clean_up_sentence(sentence):
    # tokenize the pattern s
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: bag[i] = 1
            if show_details:
                print ("found in bag: %s" % w)

    return(np.array(bag))

p = bow("is your shop open today?", words)
#print (p)
#print (classes)

def deserialize_json(cls=None, path=None):
    def read_json(_path):
        with open(_path, "r") as file:
            return json.load(file)

    data = read_json(path)
    ret_json = json.dumps(data)
    return ret_json

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # import the trained model
    model.load('./model.tflearn')
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def response(sentence, userID='123', show_details=False):
    returnText = "Did you want me to look up following Intents for this: "
    bestPossibleIntent = ""
    identifiedContexts = []
    results = classify(sentence)
    if os.path.isdir("c:\\MachineLearning_Projects\\processed") and results[0][0] == "unknown":
        name_list = os.listdir("c:\\MachineLearning_Projects\\Intents")
        full_list = [os.path.join("c:\\MachineLearning_Projects\\Intents",i) for i in name_list]
        time_sorted_list = sorted(full_list, key=os.path.getmtime,  reverse=True)
        for timesortedFile in time_sorted_list:
            intentFileName = os.path.splitext(timesortedFile)[0]
            d = deserialize_json(ResolvedContext, intentFileName + ".json")
            resolvedContexts.append(d)
            resolvedContext = ResolvedContext(d)
            identifiedContexts.append(resolvedContext.lastintent)
        for contextFound in resolvedContexts:
            print("Found following contexts: " + contextFound)
        results.append(resolvedContext.lastintent)

    global lastintent
    global userContext
    global files
    print("Result Size: " + str(len(resolvedContexts)))
    # if we have a classification then find the matching intent tag
    if len(resolvedContexts) == 0:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    if  i['tag'] != "unknown":
                        userContext.lastintent = i['tag']
                        userContext.namedentity = "services"
                        if len(i['nextactions']) != 0:
                            userContext.nextaction = i['nextactions']
                        fileName = "c:\\MachineLearning_Projects\\Intents\\" + userContext.lastintent + ".json" 
                        serialize_json(userContext, fileName)
                        files.append(fileName)
                        with open("c:\\MachineLearning_Projects\\processed\\" + userContext.lastintent + ".txt", "w") as file:
                            file.write("Your text goes here")
                    # a random response from the intent
                    return random.choice(i['responses'])

            results.pop(0)
    else:
         while identifiedContexts:
            possibleIntent = identifiedContexts.pop(0)
            if bestPossibleIntent == "":
                bestPossibleIntent = possibleIntent
            returnText = returnText + possibleIntent + ", " 
         returnText = returnText + "Based on my analysis following Action makes most sense to look up: " + bestPossibleIntent
         return returnText

if args.question:
    print (response(args.question))
