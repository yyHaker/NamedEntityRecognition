# -*- coding: utf-8 -*-
import numpy as np
import codecs
import gensim
from gensim.models.keyedvectors import KeyedVectors

word_embedding_file = './resource/news12g_bdgbk20g_nov90g_dim128.txt'
#fw = codecs.open(word_embedding_file, 'rb')
#for line in fw:
#    values = line.strip()
#    print(values)
    #print(values[0], values[1])

# model = gensim.models.Word2Vec.load(word_embedding_file)
# print(model.most_similar([u'李连杰', u'基金'], [u'成龙']))
# word2vectors = KeyedVectors.load_word2vec_format(word_embedding_file, binary=True)
# word2vectors.most_similar_to_given("但是")
# print(word2vectors)
# word2vectors.save_word2vec_format('./resource/news12g_bdgbk20g_nov90g_dim128.txt', binary=False)

f = open(word_embedding_file, 'r', encoding='utf-8')
for index, line in enumerate(f):
    if index == 0:
        continue    # 跳过第一行
    if index > 10:
        break
    values = line.split()
    try:
        coefs = np.asarray(values[1:], dtype='float32')  # 取向量
    except ValueError:
        # 如果真的这个词出现在了训练数据里，这么做就会有潜在的bug。那coefs的值就是上一轮的值。
        #print(values[0], values[1:])
        print("error found")
        break

    # embeddings[index] = coefs   # 将词和对应的向量存到字典里
    print(values[0], values[1:])
    # print("the embedding size is", coefs.shape)
    # print(type(values[0]))
    # print(type(values[0]) is str)
f.close()
print("结束")
