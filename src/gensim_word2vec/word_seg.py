# -*-  coding: utf-8 -*-
"""
文本数据分词(将中文分解为一个一个的字)
"""
import jieba
import codecs
import os
import six
from get_raw import Config


def is_alpha(tok):
    """判断tok使用ascii编码后是否是alpha"""
    try:
        return tok.encode('ascii').isalpha()
    except UnicodeEncodeError:
        return False


def zhwiki_segment(_config, remove_alpha=True):
    """中文分词"""
    i = 0
    if not six.PY3:
        output = open(os.path.join(_config.data_path, _config.zhwiki_seg_t2s), 'w', encoding='utf-8')
    output = codecs.open(os.path.join(_config.data_path, _config.zhwiki_seg_t2s), 'w', encoding='utf-8')
    print('Start...')
    with codecs.open(os.path.join(_config.data_path, _config.zhwiki_raw_t2s), 'r', encoding='utf-8') as raw_input:
        for line in raw_input.readlines():
            line = line.strip()
            i += 1
            print('line ' + str(i))  # 多少行？
            text = line.split()  # sent list
            if True:
                text = [w for w in text if not is_alpha(w)]   # 去掉英文?
            word_cut_seed = [jieba.cut(t) for t in text]
            tmp = ''
            for sent in word_cut_seed:
                for tok in sent:
                    tmp += tok + ' '
            tmp = tmp.strip()
            if tmp:
                output.write(tmp + '\n')
        output.close()


def zhwiki_word_segment(_config, remove_alpha=True):
    """将中文分割为字符"""
    i = 0
    if not six.PY3:
        output = open(os.path.join(_config.data_path, _config.zhwiki_seg_t2s), 'w', encoding='utf-8')
    output = codecs.open(os.path.join(_config.data_path, _config.zhwiki_seg_t2s), 'w', encoding='utf-8')
    print('Start...')
    with codecs.open(os.path.join(_config.data_path, _config.zhwiki_raw_t2s), 'r', encoding='utf-8') as raw_input:
        for line in raw_input.readlines():
            line = line.strip()
            i += 1
            print('line ' + str(i))  # 多少行？
            text = line.split()  # sent list
            if True:
                text = [w for w in text if not is_alpha(w)]   # 去掉英文?
            word_cut_seed = [list(sent) for sent in text]  # 分割为中文字符
            tmp = ''
            for sent in word_cut_seed:
                for tok in sent:
                    tmp += tok + ' '
            tmp = tmp.strip()
            if tmp:
                output.write(tmp + '\n')
        output.close()


config = Config()
zhwiki_word_segment(config)