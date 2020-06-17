# 标注数据
#

import numpy as np
from test import cut_sent


# from kashgari.tasks.labeling import BiLSTM_Model
# import matplotlib.pyplot as plt

def load_entity_dict(entity_dict_file_path):
    """
    加载实体词典
    :param entity_dict_file_path:
    :return:
    """
    dict_file = open(entity_dict_file_path, encoding='utf-8')
    entity_lines = dict_file.read().splitlines()
    entity_dict = {}
    for entity_line in entity_lines:
        words = entity_line.split('\t')
        entity_dict[words[0]] = words[1]
    entity_dict_keys = sorted(entity_dict.keys(), reverse=True)
    return entity_dict, entity_dict_keys


def label_sentence(sentence, entity):
    tag = ['O'] * len(sentence)
    mark = [0] * len(sentence)
    if entity.name in sentence:
        print(sentence)
        index = sentence.find(entity.name)
        tag[index] = 'B-' + entity.type
        for j in range(1, len(entity.name)):
            tag[index + j] = 'I-' + entity.type
    return tag


def label_sentence_with_entity_dict(sentence, entity_dict, entity_dict_sorted_entity_name):
    """
    利用术语词典标注句子
    :param entity_dict_sorted_entity_name:
    :param sentence:
    :param entity_dict: 术语字典 {'包装机': '2', '包装机收尘器': '2', '回转窑': '2'}
    :return:
    """
    tag_label = ['O'] * len(sentence)
    mark_lable = [0] * len(sentence)
    # entities = entity_dict.keys()
    for entity in entity_dict_sorted_entity_name:
        entity_index = sentence.find(entity)
        if entity_index >= 0:
            entity_len = len(entity)
            if mark_lable[entity_index] == 0 and mark_lable[entity_index + entity_len - 1] == 0:
                tag_label[entity_index] = 'B-' + entity_dict[entity]
                for kk in range(1, entity_len):
                    tag_label[entity_index + kk] = 'I-' + entity_dict[entity]
                mark_lable[entity_index] = 1
                mark_lable[entity_index + entity_len - 1] = 1

    return tag_label


# def plot_graphs(history, string):
#   plt.plot(history.history[string])
#   plt.plot(history.history['val_'+string])
#   plt.xlabel("Epochs")
#   plt.ylabel(string)
#   plt.legend([string, 'val_'+string])
#   plt.show()

def get_train_data(file_path):
    """
    测试文档
    """
    # file_path = 'test.txt'
    file = open(file_path, encoding='utf-8')
    lines = file.read().splitlines()

    ed, kess = load_entity_dict('dict/cement_term_dictionary.txt')
    print(len(kess))
    print(kess)
    chars = []
    tags = []
    for line in lines:
        sentences = cut_sent(line)
        for sentence in sentences:
            tag = label_sentence_with_entity_dict(sentence, ed, kess)
            chars.append(list(sentence))
            tags.append(tag)

    print(len(chars))
    print(len(tags))
    print(chars[1])
    print(tags[1])
    print(len(chars[1]))
    print(len(tags[1]))
    train_len = 4 * len(chars) // 5

    train_x = chars[:train_len]
    test_x = chars[train_len:]
    train_y = tags[:train_len]
    test_y = tags[train_len:]
    print("train_x le:n", len(train_x))
    print("test_x le:n", len(test_x))
    return train_x, train_y, test_x, test_y


def parse_data():
    return


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = get_train_data()  # 测试

    """
    cement.txt 总共9120个句子.
    训练集和测试集合 4：1分割
    
    训练集合7296,测试集1824
    """
    # train_len = 7296
    # file_path = 'cement.txt'
    # file = open(file_path, encoding='utf-8')
    # lines = file.read().splitlines()
    #
    # ed, kess = load_entity_dict('dict/cement_term_dictionary.txt')
    # print(len(kess))
    # print(kess)
    #
    #
    # #test_sentence = '利用低场核磁共振的方法研究了不同水灰比的水泥浆体早期水化过程中可蒸发水量的变化。'
    # # print(kess)
    # # print(ed.keys())
    # # tagg = label_sentence_with_entity_dict(test_sentence, ed)
    # # print(test_sentence)
    # # print(tagg)
    # chars = []
    # tags = []
    # for line in lines:
    #     sentences = cut_sent(line)
    #     for sentence in sentences:
    #         tag = label_sentence_with_entity_dict(sentence, ed, kess)
    #         chars.append(list(sentence))
    #         tags.append(tag)
    #
    # print(len(chars))
    # print(len(tags))
    # print(chars[1])
    # print(tags[1])
    # print(len(chars[1]))
    # print(len(tags[1]))
    #
    # train_x = chars[:train_len]
    # test_x = chars[train_len:]
    # train_y = tags[:train_len]
    # test_y = chars[train_len:]
    # print("train_x le:n", len(train_x))
    # print("test_x le:n", len(test_x))
    # model = BiLSTM_Model()
    # history = model.fit(train_x, train_y, test_x, test_y, epochs=15)
    # plot_graphs(history, "acc")
    # plot_graphs(history, "loss")
    # print('validata model')
    # model.evaluate(test_x, test_y)

# print(ed)
# keys = ed.keys()
# print('keys', keys)
# print(type(keys))
# mark = [0] * 10
# print(mark)
# for k in keys:
#     print(k)
