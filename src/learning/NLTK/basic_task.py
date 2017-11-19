# -*- coding: utf-8 -*-
import nltk

"""
分词
"""
text = 'PythonTip.com is a very good website. We can learn a lot from it.'
# 将文本拆分成句子列表
sens = nltk.sent_tokenize(text)
print sens
# ['PythonTip.com is a very good website.', 'We can learn a lot from it.']
# 对句子进行分词，nltk的分词是句子级的，因此要先分句，再逐句分词，否则效果会很差
words = []
for sent in sens:
    words.append(nltk.word_tokenize(sent))
print words
# [['PythonTip.com', 'is', 'a', 'very', 'good', 'website', '.'],
# ['We', 'can', 'learn', 'a', 'lot', 'from', 'it', '.']]

"""
词性标注
"""
tags = []
# 词性标注要利用上一步分词的结果
for tokens in words:
    tags.append(nltk.pos_tag(tokens))
print tags
# [[('PythonTip.com', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('very', 'RB'), ('good', 'JJ'),
# ('website', 'NN'), ('.', '.')], [('We', 'PRP'), ('can', 'MD'), ('learn', 'VB'), ('a', 'DT'),
# ('lot', 'NN'), ('from', 'IN'), ('it', 'PRP'), ('.', '.')]]

"""
命名实体识别
"""
text = "Xi is the chairman of China in the year 2013."
# 分词
tokens = nltk.word_tokenize(text)
# 词性标注
tags = nltk.pos_tag(tokens)
print tags
# [('Xi', 'NN'), ('is', 'VBZ'), ('the', 'DT'), ('chairman', 'NN'),
# ('of', 'IN'), ('China', 'NNP'), ('in', 'IN'), ('the', 'DT'), ('year', 'NN'),
# ('2013', 'CD'), ('.', '.')]
# NER需要利用词性标注的结果
ners = nltk.ne_chunk(tags)
print '%s --- %s' % (str(ners), str(ners.label()))
"""
(S
  (GPE Xi/NN)
  is/VBZ
  the/DT
  chairman/NN
  of/IN
  (GPE China/NNP)
  in/IN
  the/DT
  year/NN
  2013/CD
  ./.) --- S
"""

"""
句法分析
"""


