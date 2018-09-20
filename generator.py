# A Text Generator AI that generates a text based on a previously learned text
# This is no model where classification accuracy is the optimization problem.
# This algorithm tries to generalize the dataset and generate new text.
# See: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
# author: Tobias Freundorfer

import numpy
import datetime
# Class that handles everything regarding Files (e.g. saving or loading objects)
from filehelper import FileHelper
# Class that handles the Preprocessing phase
from preprocessing import Preprocessing
# Class that handles the Training phase
from training import Training
# Imports for the NN model used
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# The length of a the sequence to use as input length for training
#TRAINING_SEQ_LENGTH = 100
# Whether the data necessary for training should be loaded from previous checkpoints (= True) or calculated (= False)
#LOAD_TRAININGDATA_FROM_CHECKPOINTS = True
# Whether previously trained weights should be used (= True) or new weights should be trained
#LOAD_TRAINING_WEIGHTS_FROM_CHECKPOINT = True
#LOAD_TRAINING_WEIGHTS_FROM_CHECKPOINT_FILE = 'trainingCheckpoints/weights-ep_02-loss_2.7879.hdf5'

# The length of the text that should be generated (in chars)
#GENERATION_TEXT_LENGTH = 1000


def generate_random_seed(X):
    """Generates a random seed used as starting value from the text generation.
    The seed is a sequence from the training data used as a starting point.
    In this case it's a random sequence.
    """
    startIndex = numpy.random.randint(0, len(X)-1)
    seed = X[startIndex]

    return seed


def generate_text(desiredTextLength, index2charDict, vocabulary, seed, model):
    """Generates text based on the learned text
    """
    print('Starting Text-Generation Phase...')
    start = datetime.datetime.now()
    text = ""
    # Predict the next chars starting from the seed
    for i in range(desiredTextLength):
        x = numpy.reshape(seed, (1, len(seed), 1))
        x = x / float(len(vocabulary))
        # TODO Could deactivate verbose here
        prediction = model.predict(x, verbose=1)
        # Find the prediction with the highest probability
        predictedIndex = numpy.argmax(prediction)
        predictedChar = index2charDict[predictedIndex]
        text += str(predictedChar)
        # Append the new predicted char to the seed and repredict the following sequence
        seed.append(predictedIndex)
        # Move the window by one character
        seed = seed[1:len(seed)]

    end = datetime.datetime.now()
    deltaTime = end-start
    print('Generation finished: %ds' % deltaTime.total_seconds())
    return text


def main():
    X = []
    Y = []
    char2indexDict = None
    index2charDict = None
    vocabulary = None
    config = FileHelper.load_config('config.json')

    seq_length = config['preprocessing']['sequence_chars_length']

    # Load data or preprocess
    if not config['preprocessing']['exec_preprocessing']:
        X = FileHelper.load_object_from_file(
            config['preprocessing']['checkpoints']['X_file'])
        Y = FileHelper.load_object_from_file(
            config['preprocessing']['checkpoints']['Y_file'])
        char2indexDict = FileHelper.load_object_from_file(
            config['preprocessing']['checkpoints']['char2indexDict_file'])
        index2charDict = FileHelper.load_object_from_file(
            config['preprocessing']['checkpoints']['index2charDict_file'])
    else:
        preprocessing = Preprocessing(config)
        X, Y, char2indexDict, index2charDict = preprocessing.preprocess()
        FileHelper.save_object_to_file(
            config['preprocessing']['checkpoints']['X_file'], X)
        FileHelper.save_object_to_file(
            config['preprocessing']['checkpoints']['Y_file'], Y)

    # TODO Good idea to always load it from file?
    vocabulary = FileHelper.load_object_from_file(
        config['preprocessing']['checkpoints']['vocabulary_file'])

    # Save the unshaped version of X because it's needed for generation later
    X_unshaped = X

    # Transform the data to the format the LTSM expects it [samples, timesteps, features]
    X = numpy.reshape(X, (len(X), seq_length, 1))
    # Normalize/rescale all integers to range 0-1
    X = X / float(len(vocabulary))
    # As usual do one-hot encoding for categorial variables to the output variables (vector of zeros with a single 1 --> 0..N-1 categories)
    Y = np_utils.to_categorical(Y)

    training = Training(config)
    # Define the model
    model = training.define_model(X, Y)

    if config['training']['exec_training']:
        # Train the model
        model = training.train(X, Y, char2indexDict, vocabulary, model)
    else:
        # Just set the previously trained weights for the model
        model.load_weights(config['training']['load_weights_filename'])
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    if config['generation']['exec_generation']:
        # Generate the random seed used as starting value for text generation
        seed = generate_random_seed(X_unshaped)
        generatedText = generate_text(
            config['generation']['text_chars_length'], index2charDict, vocabulary, seed, model)

        # Save the generated text to file
        outputFilename = config['generation']['foldername'] + '/' + \
            datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + '.txt'
        FileHelper.write_data(outputFilename, generatedText)


if __name__ == '__main__':
    main()
