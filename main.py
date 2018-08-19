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
test_data_1 = r"""
这项技术来自于世界知识产权组织授权的一项三星电子专利。在专利书中三星电子描述称，消费者经常面临着微小的屏幕划伤和指纹污迹的问题。
三星电子提出的解决方式就是在手机屏幕上创建一个能够自我修复的保护层，它也能够防止人们留下肮脏的手指印。
通常智能手机屏幕都覆盖着钢化膜。三星电子自己的手机产品配备了Corning公司打造的强化金刚玻璃。
这种金刚玻璃设计的非常坚硬，但是这并非是一种完美的解决方案。即使是三星电子也承认，这种玻璃容易打碎或者出现划伤。
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