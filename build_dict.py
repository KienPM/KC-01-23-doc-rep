""" Create by Ken at 2020 Jun 01 """
import os
import re
from pymongo import MongoClient
from tqdm import tqdm

mongo_client = MongoClient('localhost', 27017)
mongo_client.server_info()
db = mongo_client['KC_01_23']
tokenized_collection = db['tokenized']
regex = re.compile(r'[0-9@!#$%^&*()<>?/\|}{~:.,–]')


def load_stopwords(stopwords_path):
    file = open(stopwords_path)
    lines = file.readlines()
    stopwords = []
    for line in lines:
        stopword = line.strip()
        stopword = re.sub(r'\s+', '_', stopword)
        stopwords.append(stopword)
    file.close()
    return stopwords


def process_part(sentences):
    result = []
    for sentence in sentences:
        for token in re.split(r'\s+', sentence):
            token = token.lower().strip()
            token = re.sub(u"\u200B", "", token)
            if len(token) > 1 and token not in stopwords and regex.search(token) is None:
                result.append(token)

    return result


def build_dict():
    vocab = set()

    docs = tokenized_collection.find()
    for doc in tqdm(list(docs)):
        vocab.update(process_part(doc['title']))
        vocab.update(process_part(doc['content']))
        vocab.update(process_part(doc['summary']))

    vocab = list(vocab)
    vocab.sort()

    os.makedirs('data', exist_ok=True)
    output = open(os.path.join('data', 'id2word.dict'), 'w')
    for word in vocab:
        output.write(f'{word}\n')
    output.close()


if __name__ == '__main__':
    stopwords_path = '/media/ken/Data/Code/pycharm/LDA/clustering/data/stopwords.csv'
    stopwords = set(load_stopwords(stopwords_path))
    build_dict()
