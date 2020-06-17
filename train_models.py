from keras import Sequential
from keras.layers import Embedding, Bidirectional
from keras_contrib.layers import CRF

import label_data

train_x, train_y, test_x, test_y = label_data.get_train_data()
label_set = set()
for sentence in train_y:
    for y_lable in sentence:
        label_set.add(y_lable)
print(train_y[1])
print(type(train_y[1]))
print(label_set)


def build_lstm_crf_model(vocab, chunk_tags):
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    return model
