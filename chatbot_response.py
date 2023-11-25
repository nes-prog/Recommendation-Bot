from keras.models import load_model
from train_model import *
from recommendation import *

cosine_sim_author, cosine_sim_genre = create_mat_cos_sim()

def clean_up_sentence(sentence):
    "preprocess user_request (tokenization and stemming)"
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    "Encoding user_request after preprocessing"
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence):
    ""
    # filter out predictions below a threshold
    model = load_model('model_intents.h5')
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(sentence):
    "chatbot response for each user request"
    ints=predict_class(sentence)
    if ints[0]['intent']=='Asking about the most rated books': 
        message=top_rated('book_rating')
        return message
    elif (ints[0]['intent']=='asking about similar books ') and ('author' in sentence):
        message=get_recommendation_author_or_genre(find_title_content_based(sentence), cosine_sim_author)
        return message
    elif (ints[0]['intent']=='asking about similar books ') and ('genre' in sentence):
        message=get_recommendation_author_or_genre(find_title_content_based(sentence), cosine_sim_genre)
        return message
    elif ints[0]['intent']=="asking about book's recommendation based on user's preference":
        message = get_recommendation_collaborative(find_title_content_based(sentence), bookmat)
        return message
    elif ints[0]['intent']=="Asking about informations":       
        message = asking_about_information(find_title_content_based(sentence))
        return message
    else:
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if(i['tag']== ints[0]['intent']):
                result = random.choice(i['responses'])
                message={ "answer":[{"_type": "dialog",
                    "message": result}]}
        return message

