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
def split_word(sentence: str):
    words = word_tokenize(sentence)
    words = [word for word in words if word.isalpha()]  # コンマやピリオドを除外
    # numpy arrayに変換
    words = np.array(words)
    # 重複を除く
    words_unique = np.unique(words)
    return words, words_unique

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

# co-occurrence matrix を作成
def create_co_occurrence_matrix(words: np.ndarray, word_dict: dict, index_dict: dict) -> np.ndarray:
    len = word_dict.__len__()

    # matrixを初期化
    matrix = np.zeros((len, len))

    # 行列を作成
    for i in range(words.size - 1):
        # i, i+1番目の2つの単語に着目
        w1 = words[i]
        w2 = words[i+1]

        index1 = index_dict[w1]
        index2 = index_dict[w2]

        matrix[index1][index2] += 1
        matrix[index2][index1] += 1

        print(f"i: {i}, w1: {w1}, w2: {w2}, index1: {index1}, index2: {index2}, matrix[index1][index2]: {matrix[index1][index2]}, matrix[index2][index1]: {matrix[index2][index1]}\n")

    return matrix

# co-occurence matricsを可視化
def print_matrics(matrics: np.ndarray, file: str):
    rows, columns = matrics.shape
    print(f"row: {rows}, column: {columns}\n")

    with open(file, 'w') as file:
        for i in range(rows):
            for j in range(columns):
                file.write(f"{matrics[i][j]}, ")
            file.write("\n")


# 標準出力の省略をなくす
np.set_printoptions(threshold=100000)
input_corpus = file_to_str('/Users/s15255/work/deep-learning-from-scratch-2/ch02_/fetched_corpus_data.txt')
words, words_unique = split_word(input_corpus)
word_dict = create_index_to_word_dict(words_unique)
index_dict = create_word_to_index_dict(word_dict)
co_occurrence_matrix = create_co_occurrence_matrix(words, word_dict, index_dict)


print_matrics(co_occurrence_matrix, '/Users/s15255/work/deep-learning-from-scratch-2/ch02_/matrics')
