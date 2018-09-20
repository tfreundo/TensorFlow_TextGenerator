from filehelper import FileHelper
import datetime

class Preprocessing:
    
    def __init__(self, config):
        self.config = config

    def generate_char_to_int_dictionary(self, data):
        """Creates the mapping of unique characters to unique integers
        """
        # Sets are unordered collections of unique elements
        charSet = set(data)
        # Put the set into a list and sort it
        chars = list(charSet)
        vocab = sorted(chars)
        # FileHelper.save_object_to_file('preprocessingCheckpoints/vocab', vocab)
        FileHelper.save_object_to_file(
            self.config['preprocessing']['checkpoints']['vocabulary_file'], vocab)

        chars_len = len(data)
        vocab_len = len(vocab)
        print('Input data consists of %d Total Characters and a Vocabular of %d Characters' % (
            chars_len, vocab_len))
        return dict((character, index) for index, character in enumerate(vocab))


    def generate_int_to_char_dictionary(self, data):
        """Creates the mapping of unique integers to unique characters
        """
        # Sets are unordered collections of unique elements
        charSet = set(data)
        # Put the set into a list and sort it
        chars = list(charSet)
        vocab = sorted(chars)
        return dict((index, character) for index, character in enumerate(vocab))


    def clean_data_map(self, map):
        """Cleans the data map from unwanted characters to further improve 
        """
        # TODO Remove unwanted characters from map and reevaluate how well the algorithm performs afterwards
        return map


    def generate_training_patterns(self, data, char2intDict, saveToFile=False):
        """Generates the mapping between the input and output pairs enconded as integers.
        The length of the sequence is used to determine the first sequence, afterwards the window is sliding one index further (always with window size = length of sequence)
        """
        # The input
        X = []
        # The output
        Y = []

        seq_length = self.config['preprocessing']['sequence_chars_length']

        # for i in range(0, len(data) - TRAINING_SEQ_LENGTH, 1):
        for i in range(0, len(data) - seq_length, 1):
            # Get the text sequence of the desired length
            seq_input = data[i:i+seq_length]
            # The next character after the sequence
            seq_output = data[i + seq_length]

            # Append the integers for each char in the input sequence to X
            X.append([char2intDict[character] for character in seq_input])
            # Append the according integer of the next character that succeeds the input sequence
            Y.append(char2intDict[seq_output])

        print('Generated %d Patterns from data' % len(X))

        # TODO Save to file so that I don't need to recalculate this over and over again

        return X, Y


    def preprocess(self):
        """Executes the preprocessing which generates the data used for learning
        """
        print('Starting Preprocessing Phase...')
        start = datetime.datetime.now()
        raw_data = FileHelper.read_data_lower(
            self.config['preprocessing']['input_file'])
        # Model the characters as integers
        char2intDict = self.generate_char_to_int_dictionary(raw_data)
        # FileHelper.save_object_to_file('preprocessingCheckpoints/char2indexDict', char2indexDict)
        FileHelper.save_object_to_file(
            self.config['preprocessing']['checkpoints']['char2intDict_file'], char2intDict)
        int2CharDict = self.generate_int_to_char_dictionary(raw_data)
        FileHelper.save_object_to_file(
            self.config['preprocessing']['checkpoints']['int2charDict_file'], int2CharDict)

        # Generate the text patterns
        X, Y = self.generate_training_patterns(raw_data, char2intDict)
        end = datetime.datetime.now()
        deltaTime = end-start
        print('Preprocessing finished: %ds' % deltaTime.total_seconds())

        return X, Y, char2intDict, int2CharDict
