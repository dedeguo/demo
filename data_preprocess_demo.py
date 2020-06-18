from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy
import numpy as np
import os
import time
import label_data
import train_models

chunk_tags = ['O', 'B-1', 'I-1', 'B-2', 'I-2', 'B-3', 'I-3', 'B-4', 'I-4', 'B-5', 'I-5', 'B-6', 'I-6', 'B-7', 'I-7']


def fun1(path_to_file):
    # path_to_file = tf.keras.utils.get_file('shakespeare.txt',
    #                                        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    # path_to_file = 'test.txt'
    # 读取并为 py2 compat 解码
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # 文本长度是指文本中的字符个数
    print('Length of text: {} characters'.format(len(text)))

    # 看一看文本中的前 250 个字符
    print(text[:250])

    # 文本中的非重复字符
    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))

    # 创建从非重复字符到索引的映射
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])
    print(text_as_int)
    return char2idx, idx2char, vocab


def _process_data_x(data_x, vocab, maxlen=None):
    """

    :return:
    """
    if maxlen is None:
        maxlen = max(len(s) for s in data_x)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data_x]  # set to <unk> (index 1) if not in vocab
    x = pad_sequences(x, maxlen)
    return x


def _process_data_y(data_y, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data_y)
    y_chunk = [[chunk_tags.index(w) for w in s] for s in data_y]

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return y_chunk


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def plot_graphs(history_data, string):
    plt.plot(history_data.history[string])
    plt.plot(history_data.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()



if __name__ == "__main__":
    #file_to_pt = './data/ccement.txt'
    file_to_pt = './data/all_cement.txt'
    char2idx1, idx2char1, vocab1 = fun1('./cement.txt')
    train_x, train_y, test_x, test_y = label_data.get_train_data(file_to_pt)
    processed_x = _process_data_x(train_x, vocab1)
    processed_y = _process_data_y(train_y, chunk_tags)
    processed_test_x = _process_data_x(test_x, vocab1)

    for tt in test_y:
        for t in tt:
            if t not in chunk_tags:
                print(tt)
    processed_test_y = _process_data_y(test_y, chunk_tags)


    # print(processed_x[336])
    # print(type(processed_x[336]))
    # print(processed_y[336])
    # print(type(processed_y[336]))
    #
    # print('train_x[333]', train_x[336])
    # print('train_y[333]', train_y[336])


    EPOCHS = 5
    print('print len(vocab1)',len(vocab1))
    model = train_models.build_lstm_crf_model(vocab1, chunk_tags)
    # train model
    history = model.fit(processed_x, processed_y, batch_size=128, epochs=EPOCHS, validation_data=(
        processed_test_x, processed_test_y))
    model.save('model/crf.h5')
    plot_graphs(history, 'crf_viterbi_accuracy')
    plot_graphs(history, 'loss')
