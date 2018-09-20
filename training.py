import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


class Training:

    def __init__(self, config):
        self.config = config

    def define_model(self, X, Y):
        # Sequential model
        self.model = Sequential()
        # Dimensionality of the output space
        lstm_units = self.config['training']['lstm_units']
        self.model.add(LSTM(lstm_units, input_shape=(X.shape[1], X.shape[2])))
        # For a given probability (here 20%) the results are excluded from activation (reducing overfitting and improving model performance)
        self.model.add(Dropout(self.config['training']['dropout_probability']))
        # Dense/fully connected layer (every input is connected to every output by a weight)
        self.model.add(Dense(Y.shape[1], activation='softmax'))
        return self.model

    def train(self, X, Y, char2indexDict, vocabulary, config):
        """Trains the model with the desired data
        """

        print('Starting Training Phase...')
        start = datetime.datetime.now()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

        # Due to the slowness of the learning we are using callbacks that saves weights to a file if an improvement in loss was achieved
        filename = str(self.config['training']['checkpoints']['foldername']) + '/weights-ep_{epoch:02d}-loss_{loss:.4f}.hdf5'
        callbackslist = [ModelCheckpoint(
            filename, monitor='loss', verbose=1, save_best_only=True, mode='min')]

        # epochs = Number of epochs to train the model (a single epoch is an iteration over the entire X and Y data)
        # batch-size = Number of samples per gradient update (faster for higher batch-sizes) default: 32
        # callbacks = List of Keras Callbacks to call while training (after each epoch)
        epochs_qty = self.config['training']['epochs_qty']
        batchsize = self.config['training']['gradient_batch_size']
        self.model.fit(X, Y, epochs=epochs_qty, batch_size=batchsize, callbacks=callbackslist)

        # As we are just predicting the next value, there is NO TEST DATA to evaluate against!

        end = datetime.datetime.now()
        deltaTime = end-start
        print('Training finished: %ds' % deltaTime.total_seconds())
        return self.model
