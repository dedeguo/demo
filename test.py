from collections import defaultdict

import numpy as np
import re
import jieba
import nltk
from collections import Counter
import codecs

file_path = './data/ccement.txt'
stop_word_path = 'stopwords.txt'

jieba.load_userdict('cementdict.txt')
# 加载水泥词典 提高分词效果

# def cut_sent2(paragraph): i
#     sentences = re.split('(。|！|\!|\.|？|\?)', paragraph)  # 保留分割符
#     return sentences



def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{3})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")



def seg_sentence2wordlist(file_path):
    inwards = []
    count = 0
    stop_words = load_stop_word(stop_word_path)
    file = open(file_path, encoding='utf-8')
    lines = file.read().splitlines()
    for line in lines:
        for sentence in cut_sent(line):
            if '水泥熟料' in sentence:
                count = count + 1
                print(sentence)
            wds = jieba.lcut(sentence)
            for wd in wds:
                if wd not in stop_words:  # 去停用词
                    inwards.append(wd)
    print('count:{}', count)
    return inwards


def cal_wordstatic(words_list):
    # 统计词频
    # nltk.FreqDist返回一个词典，key是不同的词，value是词出现的次数
    freq_dist = nltk.FreqDist(words_list)
    freq_list = []
    num_words = len(freq_dist.values())
    for i in range(num_words):
        freq_list.append([list(freq_dist.keys())[i], list(freq_dist.values())[i]])
    freqArr = np.array(freq_list)
    return freqArr


def load_stop_word(stop_path):
    """
    加载停用词
    :param stop_path:
    :return:
    """
    stopkey = [w.strip() for w in codecs.open(stop_path, 'r', encoding='utf-8').readlines()]
    return stopkey


def cal_word_freq(word_list):
    result = Counter(word_list)
    return result


words = seg_sentence2wordlist(file_path)
static = cal_word_freq(words)
print(static)


# file = open(file_path, encoding='utf-8')
# lines = file.read().splitlines()
# for line in lines:
#     for sentence in cut_sent(line):
#         print(sentence)
