from __future__ import print_function
 
import logging
import os.path
import six
import sys
import opencc
import re
from gensim.corpora import WikiCorpus
import jieba
import train_word2vec_model

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python process_wiki.py enwiki.xxx.xml.bz2 wiki.en.text")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    cc = opencc.OpenCC('t2s')
    stopwords = stopwordslist('CNstopwords.txt')
    with open(outp, 'w',encoding='utf8') as output:
        for text in wiki.get_texts():
            temp = ' '.join(text)
            # print(temp)
            if 'zhwiki' in inp:
                temp = cc.convert(temp)
                pattern = r'\b[A-Za-z]+\b'
                temp = re.sub(pattern, ' ', temp)
                temp = re.sub('( ){2,}', ' ', temp)
                document_cut = jieba.cut(temp, cut_all=False)
                resultlist = []
                for word in document_cut:
                    if word not in stopwords:
                        if word != '\t':
                            resultlist.append(word)
                temp = ' '.join(resultlist)
                temp = re.sub('( ){2,}', ' ', temp)
            else:
                # 这里处理其他语言的停用词
                pass
            if six.PY3:
                output.write(bytes(temp, 'utf-8').decode('utf-8') + '\n')
            else:
                output.write(temp + "\n")
            i = i + 1
            # if i == 3:
            #     break
            if (i % 100 == 0):
                logger.info("Saved " + str(i) + " articles")
    logger.info("Finished Saved " + str(i) + " articles")

    logger.info("Start training model")
    train_word2vec_model.startjob(outp, 'wiki.zh.text.model', 'wiki.zh.text.vector')
    logger.info("Complete training model")
