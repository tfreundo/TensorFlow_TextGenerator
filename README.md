# TensorFlow_TextGenerator
A Text-Generator that uses TensorFlow to train a LSTM model for a given text file and generates text in that style by predicting single chars for a desired text length.
This code was created as a private project while learning machine learning concepts. Therefore not everything is perfectly implemented and fully configurable, but has enough possibilities to play around and experiment with different settings. 
Free ebooks to use as input dataset can for example be found at the awesome [Project Gutenberg](https://www.gutenberg.org/).

## Configuration
The most relevant parameters and settings are configurable via the [config.json](config.json) file.
* preprocessing: Parameters specific to the preprocessing phase
  * exec_preprocessing: Whether preprocessing should be exectued or not
  * input_file: The file to use as input data
  * sequence_chars_length: The length of the text sequence to extract the patterns from (sequence -> predicted next char)
  * checkpoints:
    * char2intDict_file: The file holding the checkpoint for the dictionary that converts chars to integers
    * int2charDict_file: The file holding the checkpoint for the dictionary that converts integers to chars
    * vocabulary_file: The file holding the extracted vocabulary (unique chars)
    * X_file: The input matrix
    * Y_file: The output matrix (next char)
* training: Parameters specific to the training phase
  * exec_training: Whether training should be exectued or not
  * load_weights_filename: If training should not be extecuted, previously trained weights are loaded from this file
  * lstm_units: Dimensionality of the output space
  * dropout_probability: The probability of a dropout between 0 and 1
  * epochs_qty: The amount of epochs to execute while training
  * batch_size: Number of samples per gradient update
  * checkpoints:
    * foldername: The folder where the weights should be stored
* generation:
  * exec_generation: Whether generation should be exectued or not
  * text_chars_length: The length of the text in chars that should be generated
  * foldername: The folder where the resulting generated text should be stored
