import nltk
from preprocess import preprocess

# nltk.download('punkt')
import nltk

# nltk.download('averaged_perceptron_tagger')
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings
from topicModel import *
import codecs

# meta = pd.read_csv('/content/drive/My Drive/Datasets/metadata.csv')
# print(meta.shape)

warnings.filterwarnings('ignore', category=Warning)

import argparse


# def model(): #:if __name__ == '__main__':

def main():
    doc = codecs.open('train _new_us.csv', 'r', 'utf-8', errors='ignore')
    meta = pd.read_csv(doc)

    # meta = pd.read_csv('train _new_us.csv',encoding='gbk')
    print(meta.shape)
    meta.head()

    # First filter by meta file, select only papers after 2020
    # meta['publish_time'] = pd.to_datetime(meta['publish_time'])
    # meta['publish_year'] = (pd.DatetimeIndex(meta['publish_time']).year)
    # meta['publish_month'] = (pd.DatetimeIndex(meta['publish_time']).month)
    # meta = meta[meta['publish_year'] == 2020]
    print(meta.shape[0], 'Papers are available after 2020 Jan 1.')

    # Count how many has abstract
    count = 0
    index = []
    for i in range(len(meta)):
        # print(i)
        if type(meta.iloc[i, 16]) == float:
            count += 1
        else:
            index.append(i)

    print(len(index), 'Paper have abstract available')

    # extract the abstract to pandas
    documents = meta.iloc[index, 16]
    documents = documents.reset_index()
    documents.drop('index', inplace=True, axis=1)

    # create data frame with all abstracts, use as input corpus
    documents['index'] = documents.index.values
    documents.head()

    print('documents   ',documents)


    #################################################分节#######################################

    method = "LDA_BERT"
    samp_size = 5000
    ntopic = 10

    # parser = argparse.ArgumentParser(description='contextual_topic_identification tm_test:1.0')

    # parser.add_argument('--fpath', default='/kaggle/working/train.csv')
    # parser.add_argument('--ntopic', default=10,)
    # parser.add_argument('--method', default='TFIDF')
    # parser.add_argument('--samp_size', default=20500)

    # args = parser.parse_args()

    data = documents  # pd.read_csv('/kaggle/working/train.csv')
    data = data.fillna('')  # only the comments has NaN's
    rws = data.Abstract
    print('rws  ',rws)
    sentences, token_lists, idx_in = preprocess(rws, samp_size=samp_size)
    print(token_lists)
    # for z in range(len(token_lists)):
    #     print(token_lists[z])

    allToken = ''
    for q in range(len(token_lists)):
        print('one_token； ', token_lists[q])
        str1= " ".join(i for i in token_lists[q])
        allToken +=str1
        allToken += '\n'
    allSentence = ''
    for m in range(len(sentences)):
        print('one_sentence； ', sentences[m])
        allSentence += sentences[m]
        allSentence += '\n'

    # with open('/content/drive/My Drive/DL/PatentClustering/sentence.txt', 'w', encoding='utf-8') as f:
    #     f.write(allSentence)
    # with open('/content/drive/My Drive/DL/PatentClustering/token.txt', 'w', encoding='utf-8') as f2:
    #     f2.write(allToken)
    with open('sentence_us.txt', 'w', encoding='utf-8') as f:
        f.write(allSentence)
    with open('token_us.txt', 'w', encoding='utf-8') as f2:
        f2.write(allToken)
    # Define the topic model object
    # tm = Topic_Model(k = 10), method = TFIDF)



    # tm = Topic_Model(k=ntopic, method=method)
    # # Fit the topic model by chosen method
    # tm.fit(sentences, token_lists)
    # # Evaluate using metrics
    # with open("/kaggle/working/{}.file".format(tm.id), "wb") as f:
    #     pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)
    #
    # print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
    # print('Silhouette Score:', get_silhoulette(tm))
    # # visualize and save img
    # visualize(tm)
    # # for i in range(tm.k):
    # #     get_wordcloud(tm, m_clusteringtoken_lists, i)
    #
    # ##读取训练数据
    # meta2 = pd.read_csv('/content/drive/My Drive/DL/PatentClustering/train3.csv', encoding='gbk')
    # count = 0
    # index = []
    # for i in range(len(meta2)):
    #     # print(i)
    #     # print(type(meta.iloc[i,16])== str)
    #     if type(meta2.iloc[i, 16]) == float:
    #         count += 1
    #     else:
    #         index.append(i)
    # documents2 = meta2.iloc[index, 16]
    # documents2 = documents2.reset_index()
    # documents2.drop('index', inplace=True, axis=1)
    # # create data frame with all abstracts, use as input corpus
    # documents2['index'] = documents2.index.values

if __name__ == '__main__':
    main()