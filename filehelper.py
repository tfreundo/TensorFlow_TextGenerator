import pickle
import json

class FileHelper:

    @staticmethod
    def read_data_lower(filename, encoding='UTF-8'):
        """Reads the text data from the given file and returns the content with lower case
        """
        return open(file=filename, mode='r', encoding=encoding).read().lower()

    @staticmethod
    def write_data(filename, data):
        """Writes the text data to the given file
        """
        open(file=filename, mode='w').write(data)

    @staticmethod
    def save_object_to_file(filename, obj):
        """Saves an object to a file
        """
        with open(file=filename, mode='wb') as f:
            pickle.dump(obj, f)
        return

    @staticmethod
    def load_object_from_file(filename):
        """Loads an object from a file
        """
        with open(file=filename, mode='rb') as f:
            obj = pickle.load(f)
            return obj
        return

    @staticmethod
    def load_config(filename):
        with open(filename) as f:
            config = json.load(f)
            return config
        return ""
