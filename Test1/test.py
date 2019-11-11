import keras
from keras.preprocessing import sequence
import nltk.tokenize as tk
from auxiliary_info.sen_score_parse import item_score
from keras_data_helper.tockenizer_wid import get_tokenizer_wid
import numpy as np
import nltk


def load_word(file):
    sents_up = []
    sents_down = []
    labels = []
    aspects = []
    lines = open(file, 'r', encoding="utf8").readlines()
    for i in range(0, len(lines), 3):
        polarity = lines[i + 2].split()[0]
        if polarity == 'conflict':
            continue
        tokenizer = tk.WordPunctTokenizer()
        sentences = tokenizer.tokenize(lines[i].strip())
        aspect = tokenizer.tokenize(lines[i + 1].strip())
        aspects.append(aspect)
        sents_up.append(sentences[:sentences.index("aspect_term")] + aspect)
        sents_down.append(aspect + sentences[sentences.index("aspect_term") + 1:])

        if polarity == 'negative':
            labels.append(-1)
        elif polarity == 'neutral':
            labels.append(0)
        elif polarity == 'positive':
            labels.append(1)

    up_max_len = max([len(elem) for elem in sents_up])
    down_max_len = max([len(elem) for elem in sents_down])
    return sents_up, sents_down, aspects, labels, up_max_len, down_max_len


def load_postions(sents_up, sents_down, aspects):
    positions_up = []
    positions_down = []

    def each_sent(sent_up, sent_down, aspect):
        temp = [l + 1 for l in range(len(sent_up) - len(aspect))]
        temp.reverse()
        position_up = [str(-elem) for elem in temp] + ['0'] * len(aspect)
        position_down = ['0'] * len(aspect) + [str(l + 1) for l in range(len(sent_down) - len(aspect))]
        return position_up, position_down

    for sent_up, sent_down, aspect in zip(sents_up, sents_down, aspects):
        position_up, position_down = each_sent(sent_up, sent_down, aspect)
        positions_up.append(position_up)
        positions_down.append(position_down)

    return positions_up, positions_down


def load_sentiment_score(sents_up, sents_down, aspects):
    sts_up = []
    sts_down = []

    def get_score(w):
        if w in item_score.keys():
            return str(item_score[w])
        else:
            return '0'

    def each_sent(sent_up, sent_down, aspect):
        st_up = [get_score(w.lower()) for w in sent_up if w not in aspect] + ['0'] * len(aspect)
        st_down = ['0'] * len(aspect) + [get_score(w) for w in sent_down if w not in aspect]
        return st_up, st_down

    for sent_up, sent_down, aspect in zip(sents_up, sents_down, aspects):
        st_up, st_down = each_sent(sent_up, sent_down, aspect)
        sts_up.append(st_up)
        sts_down.append(st_down)

    return sts_up, sts_down


def load_pos(sents_up, sents_down, aspects):
    poses_up = []
    poses_down = []

    def each_sent(sent_up, sent_down, aspect):
        pos_up = [pos for word, pos in nltk.pos_tag(sent_up) if word not in aspect] + ['0'] * len(aspect)
        pos_down = ['0'] * len(aspect) + [pos for word, pos in nltk.pos_tag(sent_down) if word not in aspect]
        return pos_up, pos_down

    for sent_up, sent_down, aspect in zip(sents_up, sents_down, aspects):
        pos_up, pos_down = each_sent(sent_up, sent_down, aspect)
        poses_up.append(pos_up)
        poses_down.append(pos_down)

    return poses_up, poses_down


def load_text(file_path):
    sents = []
    positions = []
    poses = []
    sts = []
    sents_up, sents_down, aspects, labels, up_max_len, down_max_len = load_word(file_path)
    positions_up, positions_down = load_postions(sents_up, sents_down, aspects)
    poses_up, poses_down = load_pos(sents_up, sents_down, aspects)
    sts_up, sts_down = load_sentiment_score(sents_up, sents_down, aspects)
    sents.append(sents_up)
    sents.append(sents_down)
    positions.append(positions_up)
    positions.append(positions_down)
    poses.append(poses_up)
    poses.append(poses_down)
    sts.append(sts_up)
    sts.append(sts_down)

    return sents, positions, poses, sts, aspects, labels, up_max_len, down_max_len


def load_padding(tokenizer, up, down, up_max_len, down_max_len):
    up_seq = tokenizer.texts_to_sequences(up)
    down_seq = tokenizer.texts_to_sequences(down)
    up_pad = sequence.pad_sequences(up_seq, maxlen=up_max_len, padding='post', truncating="post")
    down_pad = sequence.pad_sequences(down_seq, maxlen=down_max_len, padding='post', truncating="post")
    return up_pad, down_pad


def load_all(train_path, test_path):
    train_data = {}
    test_data = {}
    all_word_index = {}

    train_sents, train_positions, train_poses, train_sts, \
    train_aspects, train_labels, train_up_max_len, train_down_max_len = load_text(train_path)
    test_sents, test_positions, test_poses, test_sts, \
    test_aspects, test_labels, test_up_max_len, test_down_max_len = load_text(test_path)

    train_data['tr_labels'] = keras.utils.to_categorical(train_labels, num_classes=3)
    test_data['te_labels'] = keras.utils.to_categorical(test_labels, num_classes=3)

    word_tokenizer, word_index = get_tokenizer_wid(train_sents[0], train_sents[1], test_sents[0], test_sents[1])
    position_tokenizer, position_index = get_tokenizer_wid(train_positions[0], train_positions[1], test_sents[0],
                                                           test_positions[1])
    poses_tokenizer, poses_index = get_tokenizer_wid(train_poses[0], train_poses[1], test_poses[0], test_poses[1])
    sts_tokenizer, sts_index = get_tokenizer_wid(train_sts[0], train_sts[1], test_sts[0], test_sts[1])

    all_word_index["word_index"] = word_index
    all_word_index["position_index"] = position_index
    all_word_index["poses_index"] = poses_index
    all_word_index["sts_index"] = sts_index

    train_data['tr_sents_up'], train_data['tr_sents_down'] = load_padding(word_tokenizer,
                                                                          train_sents[0],
                                                                          train_sents[1],
                                                                          train_up_max_len,
                                                                          train_down_max_len)
    test_data['te_sents_up'], test_data['te_sents_down'] = load_padding(word_tokenizer,
                                                                        test_sents[0],
                                                                        test_sents[1],
                                                                        test_up_max_len,
                                                                        test_down_max_len)

    train_data['tr_positions_up'], train_data['tr_positions_down'] = load_padding(position_tokenizer,
                                                                                  train_positions[0],
                                                                                  train_positions[1],
                                                                                  train_up_max_len,
                                                                                  train_down_max_len)
    test_data['te_positions_up'], test_data['te_positions_down'] = load_padding(position_tokenizer,
                                                                                test_positions[0],
                                                                                test_positions[1],
                                                                                test_up_max_len,
                                                                                test_down_max_len)

    train_data['tr_pos_up'], train_data['tr_pos_down'] = load_padding(poses_tokenizer,
                                                                      train_poses[0],
                                                                      train_poses[1],
                                                                      train_up_max_len,
                                                                      train_down_max_len)
    test_data['te_pos_up'], test_data['te_pos_down'] = load_padding(poses_tokenizer,
                                                                    test_poses[0],
                                                                    test_poses[1],
                                                                    test_up_max_len,
                                                                    test_down_max_len)

    train_data['tr_sts_up'], train_data['tr_sts_down'] = load_padding(sts_tokenizer,
                                                                      train_sts[0],
                                                                      train_sts[1],
                                                                      train_up_max_len,
                                                                      train_down_max_len)
    test_data['te_sts_up'], test_data['te_sts_down'] = load_padding(sts_tokenizer,
                                                                    test_sts[0],
                                                                    test_sts[1],
                                                                    test_up_max_len,
                                                                    test_down_max_len)

    return train_data, test_data, all_word_index


# train_data, test_data, all_word_index \
#     = load_all(restaurant_path[restaurant_train_path], restaurant_path[restaurant_test_path])

def test1():
    a = np.array([[1, 3, 4, 6]])
    b = np.array([[2, 4, 5, 8]])
    return a + b
