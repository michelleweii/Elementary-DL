import tensorflow as tf
# 1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 按照词频顺序为每个词汇分配一个编号，然后将词汇表保存到一个独立的vocab文件中。
import  codecs
import collections
from operator import itemgetter

RAW_DATA = 'PTB/data/ptb.train.txt'   #训练集数据文件
VOCAB_OUTPUT = 'PTB/generate/ptb.vocab'

counter = collections.Counter()  # 统计单词出现频率
with codecs.open(RAW_DATA,'r','utf-8') as f:
    for line in f:
        for word in line.strip().split():
            # .strip()移除字符串头尾指定的字符
            # .split()指定分隔符对字符串进行切片
            counter[word] += 1

# 按词频顺序对单词进行排序。
sorted_word_to_cnt = sorted(counter.items(),key=itemgetter(1),reverse=True)
# key为函数，指定取待排序元素的哪一项进行排序,True时将按降序排列.
sorted_words = [x[0] for x in sorted_word_to_cnt] # ?

# 需要在文本换行处加入句子结束符'<eos>'，这里预先将其加入词汇表。
sorted_words = ["<eos>"] + sorted_words
# 除了<eos>,还需要将<unk>和句子起始符<sos>加入词汇表，并从词汇表里删除低频词汇。
# 在PTB数据中，因为输入数据已经将低频词汇替换成了<unk>，因此不需要这一步骤。
sorted_words = ["<unk>","<sos>","<eos>"] + sorted_words
if len(sorted_words)>10000:
    sorted_words = sorted_words[:10000]

with codecs.open(VOCAB_OUTPUT,'w','utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word+"\n")






