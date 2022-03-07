import tensorflow as tf
import numpy as np
import os
import random
print("I am using tensorflow version:", tf.__version__)

# some nice constants
TEXT_VECTOR_SIZE = None
MODEL_LOOKBACK_STEP = 40 # change this to look further back in the text
TEXT_LENGTH = None

# extract some text
text = None
with open("data/text.txt") as f:
    text = f.read()
if text == None:
    raise Exception("Could not read file")
TEXT_LENGTH = len(text)

# get some nice dictionaries
dict_index = 0
letter_to_num = {}
num_to_letter = {}
for letter in text:
    if letter in letter_to_num.keys():
        pass # do nothing
    else:
        letter_to_num[letter] = dict_index
        num_to_letter[dict_index] = letter
        dict_index += 1
del dict_index
TEXT_VECTOR_SIZE = len(letter_to_num.keys())


# convert the text to a vector format --

# putting on the thinking cap ....
# TEXT_LENGTH - MODEL_LOOKBACK_STEP - 1 this needs to exist becasue the y starts a little bit in to acount for the x offset
# xshape = (textlen-lookback-1 lookback-1 vecsize)
# yshape = (textlen-lookback-1, vecsize)
X = np.zeros((TEXT_LENGTH-MODEL_LOOKBACK_STEP-1, MODEL_LOOKBACK_STEP, TEXT_VECTOR_SIZE))
Y = np.zeros((TEXT_LENGTH-MODEL_LOOKBACK_STEP-1, TEXT_VECTOR_SIZE))

# add one is to acount for the fact that the y is ahead of the last x
for i in range(TEXT_LENGTH - MODEL_LOOKBACK_STEP - 1):
    # y
    index = i + MODEL_LOOKBACK_STEP
    Y[i, letter_to_num[text[index]]] = 1
    # x
    for j in range(MODEL_LOOKBACK_STEP):
        vecindex = letter_to_num[text[index-MODEL_LOOKBACK_STEP+j]]
        X[i, j, vecindex] = 1
#cleanup cleanup everybody do their share
del text
# weird but thing that I remember you needed to do. I don't know I haven't used tensorflow in a while
X = X.reshape((-1,MODEL_LOOKBACK_STEP,TEXT_VECTOR_SIZE))
Y = Y.reshape((-1,TEXT_VECTOR_SIZE))

# print out the data
print(X.shape)
print(Y.shape)

# define the model
def create_model(layer_count, layer_nodes):
    """
    This function will create a keras sequential model with a few extra layers.
    
    args
    layer_count: controls how many dense hidden layers there are
    layer_nodes: controls how many nodes each hidden layer has

    return
    returns a really awesome keras model that you can use
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=X.shape[1:]))

    # add the layers
    for i in range(layer_count):
        model.add(tf.keras.layers.Dense(layer_nodes))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Dropout(.2))

    model.add(tf.keras.layers.Dense(TEXT_VECTOR_SIZE))
    model.add(tf.keras.layers.Activation("softmax"))
    return model

# create and ready the model
model = create_model(2, 125)
model.compile(
        optimizer="adam", # cool general-purpose optimizer
        loss="categorical_crossentropy",
        metrics=["accuracy"])

#utils
def shift(x, addable):
    """
    This just adds an element to a ndarray
    and pops of the first element
    """
    x = x.tolist()
    x.pop(0)
    x.append(addable)
    return np.array(x)

def new_length(string, length):
    """
    truncates a string by padding or removing the end of it
    """
    return string.rjust(length)

def generate(vec_input, length):
    """
    not much to say this is bad code but it generates new text with the model
    """
    output_string = ""
    for i in range(length):
        next_char_index = np.argmax(model.predict(vec_input))
        new_vec = np.zeros((TEXT_VECTOR_SIZE,))
        new_vec[next_char_index] = 1
        vec_input[0] = shift(vec_input[0], new_vec)
        output_string += num_to_letter[next_char_index]
    return output_string
    
# train the model
os.system("clear")
if False:
    model.fit(X, Y, epochs=15)
    model.save("model")
else:
    model = tf.keras.models.load_model("model")
    while True:
        # get text
        input_text = input("What is your question?\n")
        new_text = new_length(input_text, MODEL_LOOKBACK_STEP)
        vec_input = np.zeros((1, MODEL_LOOKBACK_STEP, TEXT_VECTOR_SIZE))

        # convert to vectors
        for i in range(MODEL_LOOKBACK_STEP):
            vec_input[0, i, letter_to_num[new_text[i]]] = 1
        
        # generate new text
        output_string = generate(vec_input, 100)
        print(output_string)
