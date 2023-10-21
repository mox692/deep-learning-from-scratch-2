import nltk
from nltk.tokenize import word_tokenize
import numpy as np

nltk.download('punkt')

def file_to_str(file_name: str) -> str:
    with open(file_name, 'r') as file:
        data = file.read()
    return data

# 入力文字列(結構長いやつ)を、重複のない単語の配列に分割
# https://github.com/nltk/nltk
def split_word(sentence: str) -> np.ndarray:
    words = word_tokenize(sentence)
    words = [word for word in words if word.isalpha()]  # コンマやピリオドを除外
    # numpy arrayに変換
    words = np.array(words)
    # 重複を除く
    words = np.unique(words)
    return words

# test
# print(split_word("This is a sample sentence, with some punctuation."))

# index -> string
def create_index_to_word_dict(words: np.ndarray) -> dict:
    # 単語を全てnumbering
    word_dic = {}
    for idx, v in enumerate(words):
        word_dic[idx] = v

    return word_dic

# string -> index
def create_word_to_index_dict(index_to_word_dict: dict) -> dict:
    result = {}
    for idx, v in enumerate(index_to_word_dict.values()):
        result[v] = idx

    return result

# co-occurrence matrix の雛形を作成
def create_co_occurrence_matrix(words: np.ndarray, word_dict: dict) -> np.ndarray:
    len = words.size
    matrix = np.zeros((len, len))
    return matrix

np.set_printoptions(threshold=100000)
input_corpus = file_to_str('/Users/s15255/work/deep-learning-from-scratch-2/ch02_/fetched_corpus_data.txt')
words = split_word(input_corpus)
word_dict = create_index_to_word_dict(words)
index_dict = create_word_to_index_dict(word_dict)
co_occurrence_matrix = create_co_occurrence_matrix(words, word_dict)



print(index_dict)
