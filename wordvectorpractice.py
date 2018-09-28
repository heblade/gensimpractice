"""
@author:Blade.He
@file: wordvectorpractice.py
@time: 2018/09/20
"""
import gensim
import time

# def startjobforzhwiki():
#     model = gensim.models.Word2Vec.load("./wordvectormodel/wiki.zh.text.model")
#     search = '苏格拉底'
#     print('查询词: {0}'.format(search))
#     print(model.most_similar(search))

# model: 中文wiki：./wordvectormodel/wiki.zh.text.model
# model：微信公众号：./wordvectormodel/word2vec_wx
def startjob(model='./wordvectormodel/wiki.zh.text.model'):
    start = time.time()
    model = gensim.models.Word2Vec.load(model)
    print('load word vector: {0} seconds'.format(time.time() - start))
    words = ['道琼斯', '世界杯', '落日', '天猫', '靓女', '年维泗', '搜狐', '马云', '元宵', '华南理工', '华工', '英语']
    for word in words:
        print('---------------------------------------')
        try:
            print('与单词:{0} 最接近的词有:'.format(word))
            print(model.most_similar(word))
        except Exception as e:
            print('词库中没有该词: {0}'.format(word))
        print('---------------------------------------')
        print()


if __name__ == '__main__':
    startjob()