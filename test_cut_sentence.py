import jieba
import re

jieba.add_word("中国科学院计算所")
# 输入一个段落，分成句子，可使用split函数来实现
paragraph = "生活对我们任何人来说都不容易！我们必须努力，最重要的是我们必须相信自己。 \
我们必须相信，我们每个人都能够做得很好，而且，当我们发现这是什么时，我们必须努力工作，直到我们成功。"

para2 = '小明硕士毕业于中国科学院计算所，后在日本京都大学深造'

seg = jieba.cut(para2)
print(" ".join(seg))
sentences = re.split('(。|！|\!|\.|？|\?)', paragraph)  # 保留分割符

new_sents = []
for i in range(int(len(sentences) / 2)):
    sent = sentences[2 * i] + sentences[2 * i + 1]
    new_sents.append(sent)

print(new_sents)