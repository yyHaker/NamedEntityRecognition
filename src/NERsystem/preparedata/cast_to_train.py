# -*- coding: utf-8 -*-
import codecs


with codecs.open('ner_data.txt', 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f]
    print("the total sentences are", len(sentences))  # 979个句子

# 遍历每一个句子，遍历每一个空格分开的单元，如果单元只有一个字符，标注为O，
# 如果有多个字符，标记为相应标记，最后写入到文件中
results = []
for sentence in sentences:
    sent = str(sentence).split()  # split by whitespace
    r = []
    for chars in sent:
        if len(chars) == 1:
            print(chars + " " + "O")
            r.append(chars + " " + "O")
        else:
            print(chars[0] + " " + chars[1:])
            r.append(chars[0] + " " + chars[1:])
    print()
    r.append("\n")
    results.append(r)


print("*&*"*100)
print(results)



