from keras.models import load_model, Sequential
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from intents_preprocessing import *

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def create_model( words, intent):
    "create model architecture"
    new_x_train, new_x_test, new_y_train, new_y_test = preprocess_training_data(intents)
    #architecture
    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    model = Sequential()
    model.add(Dense(128, input_shape=(len(new_x_train[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(new_y_train[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    new_hist = model.fit(np.array(new_x_train), np.array(new_y_train), epochs=200, batch_size=5, verbose=1)
    model.save('model_intents.h5', new_hist)
create_model( words, intents)