# -*- coding: utf-8 -*-
"""
使用gensim模块中的WikiCorpus获取原始文本数据
"""
from __future__ import print_function
from gensim.corpora import WikiCorpus
import jieba
import codecs
import os
import six
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing


class Config:
    data_path = 'zhwiki'
    zhwiki_bz2 = 'zhwiki-latest-pages-articles.xml.bz2'
    zhwiki_raw = 'zhwiki_raw.txt'
    zhwiki_raw_t2s = 'zhwiki_raw_t2s.txt'
    zhwiki_seg_t2s = 'zhwiki_seg.txt'
    embedded_model_t2s = 'embedding_model_t2s/zhwiki_embedding_t2s.model'
    embedded_vector_t2s = 'embedding_model_t2s/vector_t2s'


def dataprocess(_config):
    i = 0
    if not six.PY3:
        output = open(os.path.join(_config.data_path, _config.zhwiki_raw), 'w')
    output = codecs.open(os.path.join(_config.data_path, _config.zhwiki_raw), 'w')
    # cost a lot of time
    wiki = WikiCorpus(os.path.join(_config.data_path, _config.zhwiki_bz2), lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        if not six.PY3:
            output.write(b' '.join(text).decode('utf-8', 'ignore') + '\n')
        else:
            output.write(' '.join(text) + '\n')
        i += 1
        if i % 10000 == 0:
            print('Saved ' + str(i) + ' articles')
    output.close()
    print('Finished Saved ' + str(i) + ' articles')


config = Config()
dataprocess(config)