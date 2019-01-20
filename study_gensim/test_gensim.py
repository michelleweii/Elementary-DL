from gensim import corpora, models, similarities
import jieba
import re
from snownlp import SnowNLP
# 读取文本内容
f = open('data/test_gensim.txt','r',encoding='utf-8')
text = f.read()
f.close()
# 分句
s = SnowNLP(text)
text_list = s.sentences
seg_list = []
# 循环句子列表，对每个句子做分词处理
for i in text_list:
    temp_list = jieba.cut(i,cut_all=False)
    results = re.sub('[（）：:？“”《》，。！()·、.\d ]+', ' ', ' '.join(temp_list))
    seg_list.append(results)
# 将分词写入文件
f = open('data/gensim_fenci_data.txt','w',encoding='utf-8')
f.write(' '.join(seg_list))
f.close()

#…………………我是分割线………………………#
# **********字典的使用**********

# gensim的字典是将分词好的数据转换成gensim能处理的数据格式
seg_dict = [x.split(' ') for x in seg_list]
dict1 = corpora.Dictionary(seg_dict,prune_at=2000000)
print(dict1.token2id)
# 手动添加字典
dict2 = corpora.Dictionary()
dict2.token2id = {'computer': 0, 'human': 1, 'response': 2, 'survey': 3}
print(dict2.token2id)
# 合并字典
dict2 = corpora.Dictionary(seg_dict,prune_at=2000000)
dict2_to_dict1 = dict1.merge_with(dict2)
# 获取字典中某词语的词袋向量
new_doc = '人工智能 自然语言 重要'
new_vec = dict1.doc2bow(new_doc.split())
print(new_vec) # [(14, 1), (22, 1), (66, 1)] -> 14代表生态环境在字典dict1的ID，1代表出现次数
# 获取整个dict1的词袋向量
bow_corpus = [dict1.doc2bow(text) for text in seg_dict]
print(bow_corpus)

# **********字典的使用**********
#…………………我是分割线………………………#

#…………………我是分割线………………………#
# **********模型的使用**********
# 模型对象的初始化，实现词向量化
tfidf = models.TfidfModel(bow_corpus)
# 计算new_vec的权重
string_tfidf = tfidf[new_vec]
print(string_tfidf)
# 基于Tf-Idf计算相似度，参考https://radimrehurek.com/gensim/tutorial.html
index = similarities.SparseMatrixSimilarity(bow_corpus, num_features=10)
sims = index[string_tfidf]
print(sims) # 输出[(14, 0.5862218816946012), (22, 0.4809979876921243), (66, 0.6519086141926397)]
# 14代表生态环境在字典dict1的ID，0.5862218816946012代表相似性分数


# ****建模****
# 参考https://radimrehurek.com/gensim/tut2.html
# LSI建模，models.LsiModel(corpus=tfidf[bow_corpus], id2word=dict1, num_topics=50, chunksize=10000)
# HDP建模，models.HdpModel(corpus=tfidf[bow_corpus], id2word=dict1,chunksize=10000)
# RP建模，models.RpModel(corpus=tfidf[bow_corpus], id2word=dict1, num_topics=50)
lda = models.LdaModel(corpus=tfidf[bow_corpus], id2word=dict1, num_topics=50, update_every=1, chunksize=10000, passes=1)
for i in range(0, 3):
    print(lda.print_topics(i)[0])
# 利用模型获取文档的主题概率分布
doc_lda = lda[new_vec]
print(doc_lda)
# 根据模型计算相似度
# 参考https://radimrehurek.com/gensim/tut3.html
index = similarities.MatrixSimilarity(bow_corpus)
sims = index[new_vec]
print(list(enumerate(sims)))

# ****建模****
# **********模型的使用**********
#…………………我是分割线………………………#

#…………………我是分割线………………………#
# **********word2vec的使用**********
# 通过word2vec的“skip-gram和CBOW模型”生成深度学习的单词向量
# 读取已分词的文件
sentences = models.word2vec.LineSentence('data.txt')
# 建立模型，实现词向量化，第一个参数是训练语料，min_count是小于该数的单词会被踢出，默认值为5；size是神经网络的隐藏层单元数，在保存的model.txt中会显示size维的向量值。默认是100。默认window=5
model = models.word2vec.Word2Vec(sentences, size=100, window=25, min_count=5, workers=4)
# 根据语料，计算某个词的相关词列表
sim = model.wv.most_similar('生态环境', topn=10)
# 计算一个词d（或者词表），使得该词的向量v(d)与v(a="政府")-v(c="生态环境")+v(b="街道")最近
# sim = model.most_similar(positive=['政府','街道'],negative=['生态环境'], topn=10)
for s in sim:
    print("word:%s,similar:%s " %(s[0],s[1]))

# 根据语料，计算两个词的相似度 / 相关程度
print(str(model.similarity('政府','生态环境')))

# 计算文本的相似度
similarity_matrix = model.wv.similarity_matrix(dict1)
# MatrixSimilarity:指数相似性（密集与余弦距离）。
# SparseMatrixSimilarity:索引相似度（带余弦距离的稀疏）。
# SoftCosineSimilarity:指数相似性（具有软余弦距离）。
# WmdSimilarity:索引相似度（与字移动距离）。
index = similarities.SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)
sims = index[dict1.doc2bow(new_doc.split())]
print(sims)

# 保存模型方法一
model.save("test_01.model")
# 保存模型方法二
# model.wv.save_word2vec_format("test_01.model.bin",binary=True)
# model= models.KeyedVectors.load_word2vec_format("test_01.model.bin", binary=True)

# **********word2vec的使用**********
#…………………我是分割线………………………#


'''
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/7w/jy7v2r8d1xj9_3zwn5c101mw0000gn/T/jieba.cache
Loading model cost 1.252 seconds.
Prefix dict has been built succesfully.
{'一个': 0, '与': 1, '中': 2, '人工智能': 3, '处理': 4, '方向': 5, '是': 6, '的': 7, '自然语言': 8, '计算机科学': 9, '重要': 10, '领域': 11, '之间': 12, '人': 13, '各种': 14, '和': 15, '它': 16, '实现': 17, '方法': 18, '有效': 19, '理论': 20, '用': 21, '研究': 22, '能': 23, '计算机': 24, '进行': 25, '通信': 26, '一体': 27, '一门': 28, '于': 29, '数学': 30, '科学': 31, '融': 32, '语言学': 33, '因此': 34, '将': 35, '涉及': 36, '这一': 37, '人们': 38, '使用': 39, '即': 40, '日常': 41, '语言': 42, '密切': 43, '所以': 44, '有着': 45, '联系': 46, '但': 47, '区别': 48, '又': 49, '有': 50, '一般': 51, '不是': 52, '地': 53, '并': 54, '在于': 55, '研制': 56, '而': 57, '计算机系统': 58, '其中': 59, '特别': 60, '软件系统': 61, '一部分': 62, '因而': 63}
{'computer': 0, 'human': 1, 'response': 2, 'survey': 3}
[(3, 1), (8, 1), (10, 1)]
[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 2)], [(1, 1), (7, 1), (8, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1)], [(4, 1), (6, 1), (7, 1), (8, 1), (9, 1), (27, 1), (28, 1), (29, 1), (30, 1), (31, 1), (32, 1), (33, 1)], [(34, 1)], [(7, 1), (8, 1), (11, 1), (22, 1), (35, 1), (36, 1), (37, 1)], [(7, 1), (38, 1), (39, 1), (40, 1), (41, 1), (42, 1)], [(1, 1), (7, 2), (16, 1), (22, 1), (33, 1), (43, 1), (44, 1), (45, 1), (46, 1)], [(7, 1), (10, 1), (47, 1), (48, 1), (49, 1), (50, 1)], [(4, 1), (8, 2), (22, 1), (51, 1), (52, 1), (53, 1), (54, 1)], [(7, 1), (8, 1), (17, 1), (19, 1), (23, 1), (26, 1), (53, 1), (55, 1), (56, 1), (57, 1), (58, 1)], [(6, 1), (7, 1), (59, 1), (60, 1), (61, 1)], [(6, 1), (7, 1), (9, 1), (16, 1), (62, 1), (63, 1)]]
[(3, 0.7911302179382316), (8, 0.2206801933853106), (10, 0.5704500245529208)]

Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)

'''