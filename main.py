from gensim import models,corpora,similarities
import jieba.posseg as pseg
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import os

stop = [line.strip() for line in open('stopword.txt', encoding='utf-8').readlines() ]
def a_sub_b(a,b):
    ret = []
    for el in a:
        if el not in b:
            ret.append(el)
    return ret


def getdoc():
    #读取文件
    print('读取文件开始')
    raw_documents=[]
    walk = os.walk(os.path.realpath("./train"))
    for root, dirs, files in walk:
        for name in files:
            with open(os.path.join(root, name), 'r', encoding='utf-8') as cf:
                raw = str(os.path.join(root, name))+" "
                raw += cf.read()
                raw_documents.append(raw)
    print('读取文件结束')
    #构建语料库
    corpora_documents = []
    doc=[]            #输出时使用，用来存储未经过TaggedDocument处理的数据，如果输出document，前面会有u
    for i, item_text in enumerate(raw_documents):
        words_list=[]
        item=(pseg.cut(item_text))
        for j in list(item):
            words_list.append(j.word)
        words_list=a_sub_b(words_list,list(stop))
        document = TaggedDocument(words=words_list, tags=[i])
        corpora_documents.append(document)
        doc.append(words_list)
    return doc, corpora_documents


def train(corpora_documents):
    #创建model
    print('训练开始')
    model = Doc2Vec(size=50, min_count=1, window=3, workers=3)
    model.build_vocab(corpora_documents)
    model.train(corpora_documents, total_examples=model.corpus_count, epochs=70)
    model.save('./model/mymodel')
    print('训练结束')
    print('#########', model.vector_size)

model = None
doc, corpora_documents = getdoc()
if os.path.exists('./model/mymodel'):
    model = Doc2Vec.load('./model/mymodel')
else:
    train(corpora_documents)
    model = Doc2Vec.load('./model/mymodel')
#训练
# test_data_1 = r"""
# 阿森纳自从温格离开球队，桑切斯出走曼联，2018-2019赛季的英超联赛如何去打，还是个问号
# """
test_data_1 = r"""
2018雅加达亚运会小组赛最后一场比赛中，中国U23国足2比1逆转战胜阿联酋，三连胜后以小组第一昂首出线。
三场小组赛中，中国U23不但全胜，且打进11球失1球，创造了自1974年参加亚运会以来的最佳小组赛战绩。
此外在2002年和2006年的亚运会上，中国男足也曾小组赛三战全胜，但净胜球分别为9粒和4粒。
如今中国U23国足将面临新的魔咒，之前这两届小组三战全胜的比赛中，中国队出线后即被淘汰，此次新的历史继续等待他们创造。
"""
test_cut_raw_1 =[]
item2=(pseg.cut(test_data_1))
for k in list(item2):
    test_cut_raw_1.append(k.word)
inferred_vector = model.infer_vector(test_cut_raw_1)
sims = model.docvecs.most_similar([inferred_vector], topn=3)
print(sims)  #sims是一个tuples,(index_of_document, similarity)
for i in sims:
    similar=""
    print('################################')
    print('doc id: {0}, 相似度: {1}'.format(i[0], i[1]))
    for j in doc[i[0]]:
        similar+=j
    print('文本内容')
    print(similar)