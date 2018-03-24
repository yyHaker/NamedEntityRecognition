# -*- coding: utf-8 -*-
import os
import six
from get_raw import Config
from gensim.models import word2vec
import multiprocessing
import codecs


def word2vec(_config, saved=False):
    print('Start...')
    model = word2vec.Word2Vec(LineSentence(os.path.join(_config.data_path, _config.zhwiki_seg_t2s)),
                     size=50, window=5, min_count=5, workers=multiprocessing.cpu_count())
    if saved:
        model.save(os.path.join(_config.data_path, _config.embedded_model_t2s))
        model.save_word2vec_format(os.path.join(_config.data_path, _config.embedded_vector_t2s), binary=False)
    print("Finished!")
    return model


def wordsimilarity(word, model):
    semi = ''
    try:
        semi = model.most_similar(word, topn=10)
    except KeyError:
        print('The word not in vocabulary!')
    for term in semi:
        print('%s,%s' % (term[0], term[1]))


def LineSentence(path):
    """将指定路经的文本转换成iterable of iterables"""
    sentences = []
    i = 0
    with codecs.open(path, 'r', encoding="UTF-8") as raw_texts:
        for line in raw_texts.readlines():
            line = line.strip()
            sent_list = line.split()
            i += 1
            print("sent "+i)
            sentences.append(sent_list)
    print("read sentences done!")
    return sentences


config = Config()
model = word2vec(config, saved=True)
