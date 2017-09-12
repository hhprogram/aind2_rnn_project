import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    # I create a 2d array as X. each element of outer list is a subsequence of inputs
    # with length window_size. can only create length(original series) - window_size subsequences
    # though. Thus if length series was 10 and window_size was 2 then can only have 8 input
    # output pairs because the 9th input window would have valid input but no output to associate it with
    # note X is capitalized because it is a matrix
    X = [series[i:i+window_size] for i in range(len(series)- window_size)]
    # then populate the output scalars. gets the element right after each end of an input window
    # note: no need for +1 because remember the 'i+window_size'-th element in series is actually
    # one after series[i:i+window_size] because of non-inclusive slicing
    y = [series[i+window_size] for i in range(len(series) - window_size)]
    # reshape each 
    X = np.asarray(X)
    # np.shape(X) gets the shape of the nd array X. (ie if it is a 3d array with 3 rows, 4 columns and 2 other dimension)
    # then it will output a tuple of (3,4,2). So doing [0:2] just gets the first 2 dimenions (outer most dimensions)
    # and then we assign th at shape to our empty nd array X so that now it has that shape
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # build a multi-layered LSTM similar to building a normal neural network and/or CNN
    # since first layer has 5 units (or size of 5) put 5. And then the input_shape for 
    # each LSTM cell is just (window_size,1) because each cell will be reading an input
    # vector of that length as the window_size determines how many previous days' worth
    # of pricing data we use as our input for our prediction
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))
    return model



### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    # import the string module here so I can easily refer to the ascii lowercase letter
    import string
    # this creates a set of the characters found in text
    unique_chars = set(text)
    # loop through the unique characters found in text and if not found in punctuation nor 
    # ascii_lowercase then replace with a space
    for char in unique_chars:
        if char not in string.ascii_lowercase and char not in punctuation:
            text = text.replace(char, " ")

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
