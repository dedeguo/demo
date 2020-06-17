import kashgari
from kashgari.tasks.labeling import BiLSTM_CRF_Model


import data_preprocess_demo
import label_data

from data_preprocess_demo import chunk_tags, _process_data_x,_process_data_y, plot_graphs

if __name__ == "__main__":

    file_to_pt = './data/all_cement.txt'
    char2idx1, idx2char1, vocab1 = data_preprocess_demo.fun1('./cement.txt')
    train_x, train_y, test_x, test_y = label_data.get_train_data(file_to_pt)
    processed_x = data_preprocess_demo._process_data_x(train_x, vocab1)
    processed_y = data_preprocess_demo._process_data_y(train_y, chunk_tags)
    processed_test_x = data_preprocess_demo._process_data_x(test_x, vocab1)

    for tt in test_y:
        for t in tt:
            if t not in chunk_tags:
                print(tt)
    processed_test_y = _process_data_y(test_y, chunk_tags)
    print('print len(vocab1)',len(vocab1))
    print(train_x[111])
    print(train_y[111])

    model = BiLSTM_CRF_Model()
    #model.fit(processed_x,processed_y,x_validate=processed_test_x,y_validate=processed_test_y,epochs=5,batch_size=64)

    history = model.fit(train_x,train_y,test_x,test_y,epochs=15,batch_size=64)
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
