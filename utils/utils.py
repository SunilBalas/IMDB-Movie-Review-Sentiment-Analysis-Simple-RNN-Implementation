# Utility Functions
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

class Utils:
    def __init__(self):
        self.word_index, self.reverse_word_index = self.load_dataset()
    
    def load_dataset(self):
        # Load the IMDB dataset word index
        word_index = imdb.get_word_index()
        reverse_word_index = { value: key for key, value in word_index.items()}
        return word_index, reverse_word_index
    
    def decode_review(self, encoded_review):
        return ' '.join([self.reverse_word_index.get(i - 3, '?') for i in encoded_review])

    def preprocess_text(self, text):
        words = text.lower().split()
        encoded_review = [self.word_index.get(word, 2) + 3 for word in words]
        padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
        return padded_review