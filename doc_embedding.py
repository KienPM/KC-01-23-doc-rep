""" Create by Ken at 2020 Jun 07 """
import os
import argparse
import numpy as np

from gensim import corpora, models
from vncorenlp import VnCoreNLP
from underthesea import sent_tokenize

annotator = VnCoreNLP(address="http://localhost", port=9000)


def load_dict_words():
    dict_words = []
    lines = open('data/id2word.dict').readlines()
    for line in lines:
        dict_words.append(line.strip())
    return dict_words


print('Loading dictionary...')
dict_words = load_dict_words()
dictionary = corpora.Dictionary([dict_words])


def doc2words(file):
    f = open(file)
    text = f.read()
    f.close()

    doc_words = []
    for line in text.split('\n'):
        sentences = sent_tokenize(line)
        for s in sentences:
            sent_words = annotator.tokenize(s)[0]
            for w in sent_words:
                doc_words.append(w.lower())

    return doc_words


def tf_idf_doc_embedding(file):
    print('TF-IDF')
    print('Loading model...')
    model = models.TfidfModel.load('models/tf_idf/tf_idf.model')
    doc_words = doc2words(file)
    bow = dictionary.doc2bow(doc_words, allow_update=False)
    rep = model[bow]

    output_dir = os.path.dirname(file)
    output_file = os.path.basename(file).split('.')[0] + '_tf_idf.txt'
    output_path = os.path.join(output_dir, output_file)
    print(f'Writing to {output_path}...')
    out = open(output_path, 'w')
    for item in rep:
        out.write(f'{dictionary[item[0]]},{item[1]}\n')
    out.close()
    print('Done!')


def lda_doc_embedding(file):
    print('LDA')
    print('Loading model...')
    model = models.LdaModel.load('models/lda/lda.model')
    print(f'Number of topics: {model.num_topics}')
    doc_words = doc2words(file)
    bow = dictionary.doc2bow(doc_words, allow_update=False)
    rep = ['0'] * model.num_topics
    lda_output = model[bow]
    for item in lda_output[0]:
        rep[item[0]] = str(item[1])

    output_dir = os.path.dirname(file)
    output_file = os.path.basename(file).split('.')[0] + '_lda.txt'
    output_path = os.path.join(output_dir, output_file)
    print(f'Writing to {output_path}...')
    out = open(output_path, 'w')
    out.write(','.join(rep))
    out.close()
    print('Done!')


def word2vec_doc_embedding(file):
    print('word2vec')
    print('Loading model...')
    model = models.Word2Vec.load('models/word2vec/word2vec.model')
    vector_size = model.vector_size
    print(f'Vector size: {vector_size}')
    mat = []
    doc_words = doc2words(file)
    word_vectors = model.wv
    for w in doc_words:
        if w in word_vectors:
            mat.append(word_vectors[w])
        else:
            mat.append([0] * vector_size)

    mat = np.array(mat)
    rep = np.sum(mat, axis=0)
    rep /= mat.shape[0]
    rep = [str(item) for item in rep]

    output_dir = os.path.dirname(file)
    output_file = os.path.basename(file).split('.')[0] + '_word2vec.txt'
    output_path = os.path.join(output_dir, output_file)
    print(f'Writing to {output_path}...')
    out = open(output_path, 'w')
    out.write(','.join(rep))
    out.close()
    print('Done!')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Document embedding')
    arg_parser.add_argument(
        '--input_file',
        type=str,
        default='data/test_doc.txt',
        help='Path to input file'
    )
    arg_parser.add_argument(
        '--model',
        type=str,
        default='word2vec',
        help='Model to be used (tf-idf | lda | word2vec)'
    )
    args = arg_parser.parse_args()
    input_file = args.input_file
    model_type = args.model

    if model_type == 'tf-idf':
        tf_idf_doc_embedding(input_file)
    elif model_type == 'lda':
        lda_doc_embedding(input_file)
    elif model_type == 'word2vec':
        word2vec_doc_embedding(input_file)
    else:
        print(f'{model_type} is not supported! (tf-idf | lda | word2vec)')
