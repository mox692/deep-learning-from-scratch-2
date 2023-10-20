import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# https://github.com/nltk/nltk
def tokenize_sentence(sentence):
    words = word_tokenize(sentence)
    words = [word for word in words if word.isalpha()]  # コンマやピリオドを除外
    return words

sentence = "This is a sample sentence, with some punctuation."
print(tokenize_sentence(sentence))
