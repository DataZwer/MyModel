import numpy as np
import tensorflow as tf


def load_random_embedding(voc_size, dim, input_id):
    embedding_matrix = tf.Variable(tf.random_uniform([voc_size, dim], -1.0, 1.0))
    look_up = tf.nn.embedding_lookup(embedding_matrix, input_id)
    return look_up


def load_glove_embedding(word_index, file=None, trimmed_filename=None, load=False, dim=300):
    if load == True:  #
        with np.load(trimmed_filename) as data:
            return data["embeddings"]
    else:
        embeddings_index = {}
        with open(file, encoding='utf8') as f:
            for line in f:
                values = line.rstrip().split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Preparing embedding matrix.')
        embedding_matrix = np.zeros((len(word_index) + 1, dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        np.savez_compressed(trimmed_filename, embeddings=embedding_matrix)  #
    print("词向量加载完毕！")
    return embedding_matrix


# glove_file = r"D:\NLP程序相关\数据集收集\glove词向量\glove.840B.300d.txt"
# glove_embeddings_file_tw = r"D:\NLP程序相关\MyModel\data\twitter\glove_embedding_twitter.npz"
# glove_embeddings_file_res = r"D:\NLP程序相关\MyModel\data\restaurant\glove_embedding_restaurant.npz"
# glove_embeddings_file_lap = r"D:\NLP程序相关\MyModel\data\laptop\glove_embedding_laptop.npz"
# train_data, test_data, all_word_index_lap, max_len, \
# load_all(laptop_path[laptop_train_path], laptop_path[laptop_test_path])
# _, _, all_word_index_res, _, = load_all(restaurant_path[restaurant_train_path], restaurant_path[restaurant_test_path])
# _, _, all_word_index_tw, _, = load_all(twitter_path[twitter_train_path], twitter_path[twitter_test_path])
# embedding_matrix_res = down_glove(all_word_index_res[word_index], glove_file, glove_embeddings_file_res, True, 300)
# embedding_matrix_lap = down_glove(all_word_index_lap[word_index], glove_file, glove_embeddings_file_lap, True, 300)
# embedding_matrix_tw = down_glove(all_word_index_tw[word_index], glove_file, glove_embeddings_file_tw, True, 300)
