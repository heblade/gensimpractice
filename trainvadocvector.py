"""
@version: 0.1
@author: Blade He
@license: Morningstar 
@contact: blade.he@morningstar.com
@site: 
@software: PyCharm
@file: trainvadocvector.py
@time: 2018/09/21
"""
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import os
import re
import multiprocessing
import time

def getdoc(docnumber=None, outputfile='./model/vadocument.text'):
    #读取文件
    print('读取文件开始')
    raw_documents=[]
    if os.path.exists(outputfile):
        with open(outputfile, mode='r', encoding='utf-8') as raw:
            raw_documents = raw.readlines()
    else:
        walk = os.walk(os.path.realpath(r"D:\Document\automation\docs\va\txt"))
        count = 1
        isrunover = False
        for root, dirs, files in walk:
            for index, name in enumerate(files):
                print('{0} 读取文件: {1} 开始'.format(count, os.path.join(root, name)))
                with open(os.path.join(root, name), 'r', encoding='utf-8') as cf:
                    text = cf.read()
                    for temp in text.split('\n'):
                        temp = temp.replace('\0', '')
                        temp = re.sub(r'[-—•\|_\*\(\)#]', ' ', temp).strip()
                        temp = re.sub('( ){2, }', ' ', temp)
                        if len(temp.split()) > 10:
                            raw_documents.append(temp)
                print('{0} 读取文件: {1} 结束'.format(count, os.path.join(root, name)))
                if docnumber is not None and docnumber == count:
                    isrunover = True
                    break
                count += 1
            if isrunover:
                break
        print('读取所有文件结束')
    #构建语料库
    corpora_documents = []
    doc=[]
    print('构建TaggedDocument开始')
    for i, item_text in enumerate(raw_documents):
        document = TaggedDocument(words=item_text.split(), tags=[i])
        corpora_documents.append(document)
        doc.append(item_text)
    print('构建TaggedDocument结束')
    if not os.path.exists(outputfile):
        print('写入语料库开始')
        with open(outputfile, 'w', encoding='utf8') as output:
            for text in doc:
                output.write(text + "\n")
        print('写入语料库结束')
    return doc, corpora_documents


def train(corpora_documents, modelextension=''):
    #创建model
    print('训练开始')
    cores = multiprocessing.cpu_count() - 2
    models = [
        # PV-DBOW
        Doc2Vec(dm=0, vector_size=100, window=3, dbow_words=1, min_count=2, workers=cores),
        # PV-DM w/average
        Doc2Vec(dm=1, vector_size=100, window=3, dm_mean=1, min_count=2, workers=cores),
    ]
    # models = [Doc2Vec(size=100, min_count=0, alpha=0.025, min_alpha=0.025)]

    for index, model in enumerate(models):
        print('model: {0} build_vocab 开始'.format(index + 1))
        start = time.time()
        model.build_vocab(corpora_documents)
        print('model: {0} build_vocab 结束，耗时: {1} 秒'.format(index + 1, time.time() - start))
        start = time.time()
        print('model: {0} 训练开始'.format(index + 1))
        model.train(corpora_documents, total_examples=model.corpus_count, epochs=10)
        # model.save('./model/vadocmodel{0}'.format(modelextension))
        if index == 0:
            model.save('./model/vadocmodel_dbow{0}'.format(modelextension))
        else:
            model.save('./model/vadocmodel_dmaverage{0}'.format(modelextension))
        print('model: {0} 训练结束'.format(index + 1))
        print('#########', model.vector_size)
        print('训练耗时: {0} 秒'.format(time.time() - start))

def startjob():
    doc, corpora_documents = getdoc(docnumber=20, outputfile='./model/vadocument_20.text')
    train(corpora_documents, modelextension='_mini')
 
if __name__ == '__main__':
    startjob()