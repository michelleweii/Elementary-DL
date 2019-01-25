# coding:utf-8
# P235 将单词替换为词汇表中的编号
# 2
import codecs
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 将训练文件、测试文件等都根据词汇文件转化为单词编号。
# 每个单词的编号就是它在词汇文件中的行号。

RAW_DATA = 'PTB/data/ptb.valid.txt'  # 原始的训练集数据文件
VOCAB = 'PTB/generate/ptb.vocab'   # 上面生成的词汇表文件
OUTPUT_DATA = 'PTB/generate/ptb.valid'  # 将单词替换为单词编号后的输出文件

# 获取词汇表，并建立词汇到单词编号的映射。
with codecs.open(VOCAB,'r','utf-8') as f_vocab:  # 解决数据写入文件时会有编码不统一的繁琐问题，与open相比
    vocab = [w.strip() for w in  f_vocab.readlines()]
word_to_id = {k:v for (k,v) in zip(vocab,range(len(vocab)))} # 大发
print('word_to_id:',word_to_id)
# print(vocab)

# 如果出现了被删除的低频词，则替换为"<unk>"
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']

fin = codecs.open(RAW_DATA,'r','utf-8')
fout = codecs.open(OUTPUT_DATA,'w','utf-8')
for line in fin:
    words = line.strip().split()+["<eos>"]  # 读取单词并添加<eos>结束符
    # 将每个单词替换为词汇表中的编号
    out_line = ' '.join([str(get_id(w)) for w in words])+'\n'
    # 语法：  'sep'.join(seq)  上面的语法即：以sep作为分隔符，将seq所有的元素合并成一个新的字符串
    fout.write(out_line)
fin.close()
fout.close()

'''
>>> seq1 = ['hello','good','boy','doiido']
>>> print ' '.join(seq1)
hello good boy doiido
>>> print ':'.join(seq1)
hello:good:boy:doiido
'''