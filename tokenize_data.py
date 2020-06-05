""" Create by Ken at 2020 May 31 """
from pymongo import MongoClient
from tqdm import tqdm

from vncorenlp import VnCoreNLP
from underthesea import sent_tokenize

annotator = VnCoreNLP(address="http://172.27.169.164", port=9000)

mongo_client = MongoClient('localhost', 27017)
mongo_client.server_info()
db = mongo_client['KC_01_23_4_3_1']
doc_collection = db['documents']
tokenized_collection = db['tokenized']


def process_part(text):
    result = []
    for line in text.split('\n'):
        sentences = sent_tokenize(line)
        for s in sentences:
            words = annotator.tokenize(s)[0]
            result.append(' '.join(words))

    return result


def tokenize():
    docs = doc_collection.find()
    for doc in tqdm(list(docs)):
        doc['title'] = process_part(doc['title'])
        doc['content'] = process_part(doc['content'])
        doc['summary'] = process_part(doc['summary'])
        tokenized_collection.insert_one(doc)


if __name__ == '__main__':
    tokenize()
