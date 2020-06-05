""" Create by Ken at 2020 May 31 """
import csv
import re
from pymongo import MongoClient
from tqdm import tqdm

mongo_client = MongoClient('localhost', 27017)
mongo_client.server_info()
db = mongo_client['KC_01_23']
doc_collection = db['documents']
tokenized_collection = db['tokenized']


def read_data(data_file):
    csv_file = open(data_file, 'r')
    fieldnames = ("id", "title", "categories", "content", "summary")
    reader = csv.DictReader(csv_file, fieldnames)
    for d in tqdm(list(reader)):
        d['categories'] = [int(token.strip()) for token in d['categories'].split(';')]
        d['title'] = re.sub(r'\\n', '\n', d['title'])
        d['content'] = re.sub(r'\\n', '\n', d['content'])
        d['summary'] = re.sub(r'\\n', '\n', d['summary'])
        doc_collection.insert_one(d)


if __name__ == '__main__':
    data_file = '/media/ken/Data/Code/pycharm/LDA/clustering/data/QNportal_raw/post_QNPortal_all.csv'
    read_data(data_file)
