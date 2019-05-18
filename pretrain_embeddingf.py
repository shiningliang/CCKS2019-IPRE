import pkuseg
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import multiprocessing
from gensim.models import word2vec
from gensim.models.word2vec import PathLineSentences


raw_file = './data/raw/text.txt'
clean_file = './data/raw/clean_text.txt'


def stat(seq_length, type):
    print('Seq len info :')
    seq_len = np.asarray(seq_length)
    idx = np.arange(0, len(seq_len), dtype=np.int32)
    print(stats.describe(seq_len))
    plt.figure(figsize=(16, 9))
    plt.subplot(121)
    plt.plot(idx[:], seq_len[:], 'ro')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('seq_len')
    plt.title('Scatter Plot')

    plt.subplot(122)
    plt.hist(seq_len, bins=10, label=['seq_len'])
    plt.grid(True)
    plt.xlabel('seq_len')
    plt.ylabel('freq')
    plt.title('Histogram')
    plt.savefig(type + '_len_stats.jpg', format='jpg')


def clean_func(line, stopwords):
    return [word for word in line if word not in stopwords]


def clean_txt(in_path, out_path):
    with open(in_path, 'r', encoding='utf8') as fin:
        raw_lines = fin.readlines()
    fin.close()

    seg = pkuseg.pkuseg()
    seg_lines = [seg.cut(line) for line in raw_lines]
    raw_len = [len(line) for line in seg_lines]
    print('Rows of raw text - ', len(seg_lines))

    with open('./data/raw/stopwords.txt', 'r', encoding='utf8') as fs:
        stopwords = fs.readlines()
    fs.close()
    stopwords = [word.strip() for word in stopwords]

    pool = multiprocessing.Pool(processes=12)
    results = []
    for i, line in enumerate(seg_lines):
        results.append(pool.apply_async(clean_func, (line, stopwords,)))
    pool.close()
    pool.join()
    clean_lines = [res.get() for res in results]
    print('Rows of clean text - ', len(clean_lines))
    clean_lines = filter(lambda x: len(x) > 2, clean_lines)
    clean_lines = list(clean_lines)
    clean_len = [len(line) for line in clean_lines]
    print('Rows of clean text - ', len(clean_lines))

    stat(raw_len, 'raw')
    stat(clean_len, 'clean')

    with open(out_path, 'w', encoding='utf8') as fout:
        for line in clean_lines:
            fout.writelines(' '.join(line) + '\n')
    fout.close()

    print('hello world')


clean_txt(raw_file, clean_file)
