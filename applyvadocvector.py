"""
@version: 0.1
@author: Blade He
@license: Morningstar 
@contact: blade.he@morningstar.com
@site: 
@software: PyCharm
@file: applyvadocvector.py
@time: 2018/09/21
"""
import os
from gensim.models.doc2vec import Doc2Vec
from pprint import pprint
from trainvadocvector import getdoc

def startjob():
    if os.path.exists('./model/vadocmodel_dbow_mini') and \
        os.path.exists('./model/vadocmodel_dmaverage_mini'):
        doclist, corpora_documents = getdoc(docnumber=20, outputfile='./model/vadocument_20.text')

        searchtext= 'transfer your Accumulation Value'
        test_cut_raw_1 = []
        for k in searchtext.split():
            test_cut_raw_1.append(k)

        print('result from DBOW')
        modelbow = Doc2Vec.load('./model/vadocmodel_dbow_mini')
        inferred_vector = modelbow.infer_vector(test_cut_raw_1)
        simsbow = modelbow.docvecs.most_similar([inferred_vector], topn=10)
        showcontent(simsbow, doclist)

        print('result from DM Average')
        modeldm = Doc2Vec.load('./model/vadocmodel_dmaverage_mini')
        inferred_vector = modeldm.infer_vector(test_cut_raw_1)
        simsbow = modeldm.docvecs.most_similar([inferred_vector], topn=10)
        showcontent(simsbow, doclist)

def showcontent(simsbow, doclist):
    for i in simsbow:
        similar = ""
        print('################################')
        print('doc id: {0}, 相似度: {1}'.format(i[0], i[1]))
        for j in doclist[i[0]]:
            similar += j
        print('文本内容: {0}'.format(similar))

 
if __name__ == '__main__':
    startjob()