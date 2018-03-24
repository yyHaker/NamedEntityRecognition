# -*- coding: utf-8 -*-
"""
从test result中取样一些句子展示模型效果
"""
from loader import load_sentences


def transfer_sent(sent):
    """转换sent指定格式的字符串"""
    sentens = "sent:\t"
    true = "true:\t"
    predict = "pred:\t"
    for word in sent[:-1]:
        # print("word: ", word, "length: ", len(word))
        sentens += word[0] + "\t"
        true += word[1] + "\t"
        predict += word[2] + "\t"
    sentens += sent[-1][0]
    true += sent[-1][1]
    predict += sent[-1][2]
    # print()
    print(sentens)
    print(true)
    print(predict)


data_path = "current_results/F1_90.75/result/ner_predict.utf8"
sentences = load_sentences(data_path, True, True)
for sent in sentences:
    transfer_sent(sent)

#制表符的写法是\t，作用是对齐表格的各列。
print("学号\t姓名\t语文\t数学\t英语")
print("2017001\t曹操\t99\t88\t0")
print("2017002\t周瑜\t92\t45\t93")
print("2017008\t黄盖\t77\t82\t100")

