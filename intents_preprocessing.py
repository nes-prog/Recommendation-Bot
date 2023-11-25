import nltk
nltk.download('stopwords')
import string
import pickle
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import numpy as np
import random
import json
from sklearn.model_selection import train_test_split
porter = PorterStemmer()
stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()


# load intents
with open('C:/Users/Lenovo/Desktop/my_projects/freelance_work_test/Negotiation_Bot/intents.json', encoding="utf-8") as json_data:
    intents = json.load(json_data)

def create_classes_words(intents):
    "extract vocabulary (words), classes, and documents from the json file intents.json "
# Create lists
    questions = []
    words = []
    classes = []
    doc_X = []
    doc_y = []
    # browse all intentions with a For loop
    # tokenize each pattern and add the tokens to the words list, the patterns and
    # the tag associated with the intention are added to the corresponding lists
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            questions.append(pattern)
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            doc_X.append(pattern)
            doc_y.append(intent["tag"])
        
        # add tag to classes if not already there 
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
    # lemmatize all vocabulary words and convert them to lower case
    # if words don't appear in punctuation
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
    # sort vocabulary and classes alphabetically and use the
    # set to make sure there are no duplicates
    words = sorted(set(words))
    classes = sorted(set(classes))
    pickle.dump(words,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))
    pickle.dump(questions,open('questions.pkl','wb'))
    return doc_X, doc_y


def preprocess_training_data(intents):
    "Encoding the features using the bag of words"
    doc_X, doc_y= create_classes_words(intents)
    words = pickle.load(open('words.pkl','rb'))
    classes = pickle.load(open('classes.pkl','rb'))
# list for training set
    training = []
    out_empty = [0] * len(classes)
    # word set template creation
    for idx, doc in enumerate(doc_X):
        bow = []
        text = lemmatizer.lemmatize(doc.lower())
        for word in words:
            bow.append(1) if word in text else bow.append(0)
        # marks the index of the class to which the atguel pattern is associated with
        output_row = list(out_empty)
        output_row[classes.index(doc_y[idx])] = 1
        # adds the one hot encoded BoW and associated classes to the training list
        training.append([bow, output_row])
    # mix data and convert into array
    random.shuffle(training)
    training = np.array(training, dtype=object)
    # Separate features and  labels
    train_X = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))
    #split data
    new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=52)
    return new_x_train, new_x_test, new_y_train, new_y_test
