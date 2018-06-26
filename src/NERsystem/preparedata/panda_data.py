# -*- coding: utf-8 -*-
import pandas as pd
from collections import Iterable
import re
import codecs
"""
将train_data.xls中的内容那一行写入文本
"""
data = pd.read_excel('train_data.xls')
values = data["内容"].values
print(type(values))  # numpy array
# print(data["内容"].values)

# 设置分句的标志符号；可以根据实际需要进行修改
cutlist = u"。！？"


# 检查某字符是否分句标志符号的函数；如果是，返回True，否则返回False
def FindToken(cutlist, char):
    if char in cutlist:
        return True
    else:
        return False


# 进行分句的核心函数
def Cut(cutlist, lines):  # 参数1：引用分句标志符；参数2：被分句的文本，为一行中文字符
    l = []  # 句子列表，用于存储单个分句成功后的整句内容，为函数的返回值
    line = []  # 临时列表，用于存储捕获到分句标志符之前的每个字符，一旦发现分句符号后，就会将其内容全部赋给l，然后就会被清空
    if isinstance(lines, Iterable):
        for i in lines:  # 对函数参数2中的每一字符逐个进行检查 （本函数中，如果将if和else对换一下位置，会更好懂）
            if FindToken(cutlist, i):  # 如果当前字符是分句符号
                line.append(i)  # 将此字符放入临时列表中
                l.append(''.join(line))  # 并把当前临时列表的内容加入到句子列表中
                line = []  # 将符号列表清空，以便下次分句使用
            else:  # 如果当前字符不是分句符号，则将该字符直接放入临时列表中
                line.append(i)
    return l


"""
判断每一项是否是一个句子，如果是写入文本， 否则就分句写入文本，多余非完整句子的截取掉
"""
result_sents = []
for data in values:
    l = Cut(cutlist, data)
    if str(l).strip() is not None:
        # print(l)
        result_sents.extend(l)
# 打印每一个句子
# for sent in result_sents:
    #print(sent)
print("total number of sents are %d" % len(result_sents))

# 将每个句子中的字符按照空格分开, 写入到文件中
with codecs.open('ner_data.txt', 'w', encoding="utf-8") as f:
    for sent in result_sents:
        s = ""
        for char in sent:
            s = s + char + " "
        print(s)
        s = s.strip() + "\n"
        f.write(s)





