import pkuseg
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import multiprocessing
from gensim.models import word2vec
from gensim.models.word2vec import PathLineSentences
import logging
from tqdm import tqdm


raw_file = './data/raw/text.txt'
seg_file = './data/raw/seg_text.txt'
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


def clean_txt(in_path, seg_path, out_path):
    logger = logging.getLogger('Cleaning')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # logger.info('Loading raw text')
    # with open(in_path, 'r', encoding='utf8') as fin:
    #     raw_lines = fin.readlines()
    # fin.close()
    #
    logger.info('Segment raw text')
    # seg = pkuseg.pkuseg()
    # seg_lines = [seg.cut(line) for line in raw_lines]

    # pkuseg.test(in_path, seg_path, nthread=multiprocessing.cpu_count())
    logger.info('Loading seg text')
    with open(seg_path, 'r', encoding='utf8') as fin:
        seg_lines = fin.readlines()
    fin.close()

    seg_lines = [line.strip().split() for line in seg_lines]
    raw_len = [len(line) for line in seg_lines]
    logger.info('Rows of raw text - {}'.format(len(seg_lines)))
    stat(raw_len, 'raw')

    with open('./data/raw/stopwords.txt', 'r', encoding='utf8') as fs:
        stopwords = fs.readlines()
    fs.close()
    stopwords = [word.strip() for word in stopwords]

    logger.info('Removing stopwords')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = []
    for i, line in enumerate(seg_lines):
        results.append(pool.apply_async(clean_func, (line, stopwords,)))
    pool.close()
    pool.join()
    clean_lines = [res.get() for res in results]
    logger.info('Rows of clean text - {}'.format(len(clean_lines)))
    clean_lines = filter(lambda x: len(x) > 2, clean_lines)
    clean_lines = list(clean_lines)
    clean_len = [len(line) for line in clean_lines]
    logger.info('Rows of clean text - {}'.format(len(clean_lines)))
    stat(clean_len, 'clean')

    with open(out_path, 'w', encoding='utf8') as fout:
        for line in clean_lines:
            fout.writelines(' '.join(line) + '\n')
    fout.close()

    print('hello world')


clean_txt(raw_file, seg_file, clean_file)
