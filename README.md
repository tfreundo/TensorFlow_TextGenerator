# TensorFlow_TextGenerator
A Text-Generator that uses TensorFlow to train a LSTM model for a given text file and generates text in that style by predicting single chars for a desired text length.
This code was created as a private project while learning machine learning concepts. Therefore not everything is perfectly implemented and fully configurable, but has enough possibilities to play around and experiment with different settings. 
Free ebooks to use as input dataset can for example be found at the awesome [Project Gutenberg](https://www.gutenberg.org/).

## Configuration
The most relevant parameters and settings are configurable via the [config.json](config.json) file.
For most scenarios you don't actually have to edit the code yourself, just the config.

* __preprocessing__: Parameters specific to the preprocessing phase
  * __exec_preprocessing__: Whether preprocessing should be exectued or not
  * __input_file__: The file to use as input data
  * __sequence_chars_length__: The length of the text sequence to extract the patterns from (sequence -> predicted next char)
  * __checkpoints__:
    * __char2intDict_file__: The file holding the checkpoint for the dictionary that converts chars to integers
    * __int2charDict_file__: The file holding the checkpoint for the dictionary that converts integers to chars
    * __vocabulary_file__: The file holding the extracted vocabulary (unique chars)
    * __X_file__: The input matrix
    * __Y_file__: The output matrix (next char)
* __training__: Parameters specific to the training phase
  * __exec_training__: Whether training should be exectued or not
  * __load_weights_filename__: If training should not be extecuted, previously trained weights are loaded from this file
  * __lstm_units__: Dimensionality of the output space
  * __dropout_probability__: The probability of a dropout between 0 and 1
  * __epochs_qty__: The amount of epochs to execute while training
  * __batch_size__: Number of samples per gradient update
  * __checkpoints__:
    * __foldername__: The folder where the weights should be stored
* __generation__:
  * __exec_generation__: Whether generation should be exectued or not
  * __text_chars_length__: The length of the text in chars that should be generated
  * __foldername__: The folder where the resulting generated text should be stored
  
  ## Class Overview
  * __generator.py__: The class that generates text, also includes the main-function
  * __preprocessing.py__: The class that encapsulates the preprocessing phase
  * __training.py__: The class that encapsulates the training phase
  * __filehelper.py__: The helper class that handles all file specific tasks (e.g. read/write checkpoints, loading config etc.)
