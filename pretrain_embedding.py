import pkuseg
import re
import string
from zhon.hanzi import punctuation
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import multiprocessing
from gensim.models import word2vec
from gensim.models.word2vec import PathLineSentences
import logging
import os
import sys
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


def clean_func(line):
    return re.sub(r"[%s]+" % string.punctuation, "", re.sub(r"[%s]+" % punctuation, "", line))


def clean_txt(in_path, seg_path, out_path, is_clean=True, is_seg=True):
    logger = logging.getLogger('Cleaning')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if is_clean:
        logger.info('Loading raw text')
        with open(in_path, 'r', encoding='utf8') as fin:
            raw_lines = fin.readlines()
        fin.close()

        logger.info('Removing punctuations')
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = []
        for i, line in enumerate(raw_lines):
            results.append(pool.apply_async(clean_func, (line,)))
        pool.close()
        pool.join()
        clean_lines = [res.get() for res in results]
        with open(out_path, 'w', encoding='utf8') as fout:
            for line in clean_lines:
                fout.writelines(line)
        fout.close()

    if is_seg:
        logger.info('Segmenting clean text')
        # seg = pkuseg.pkuseg()
        # seg_lines = [seg.cut(line) for line in raw_lines]

        pkuseg.test(out_path, seg_path, nthread=multiprocessing.cpu_count())
        logger.info('Loading segmented text')
        with open(seg_path, 'r', encoding='utf8') as fin:
            seg_lines = fin.readlines()
        fin.close()

        seg_lines = [line.strip().split() for line in seg_lines]
        seg_len = [len(line) for line in seg_lines]
        logger.info('Rows of segmented text - {}'.format(len(seg_lines)))
        stat(seg_len, 'segmented')

        clean_lines = filter(lambda x: len(x) > 2, seg_lines)
        clean_lines = list(clean_lines)
        clean_len = [len(line) for line in clean_lines]
        logger.info('Rows of filtered text - {}'.format(len(clean_lines)))
        stat(clean_len, 'filtered')

        with open(seg_path, 'w', encoding='utf8') as fout:
            for line in clean_lines:
                fout.writelines(' '.join(line) + '\n')
        fout.close()

    print('hello world')


if __name__ == '__main__':
    # clean_txt(raw_file, seg_file, clean_file)
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    model = word2vec.Word2Vec(PathLineSentences(seg_file), sg=1, size=300, window=5, min_count=10, sample=1e-4,
                              workers=multiprocessing.cpu_count())
    model.wv.save_word2vec_format('./data/processed/word2vec.txt', binary=False)
