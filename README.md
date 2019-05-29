# A Baseline System For CCKS-2019-IPRE

## Introduction
竞赛官网：[地址](https://biendata.com/competition/ccks_2019_ipre/)
当前模型：CNN with selective attention. （由官方提供[地址](https://github.com/ccks2019-ipre/baseline)）


## Getting Started
### Environment Requirements
* python 3.6
* numpy
* tensorflow 1.12.0

### Step 1: 下载数据
Please download the data from [the competition website](https://biendata.com/competition/ccks_2019_ipre/data/), then unzip files and put them in `./data/` folder.

### Step 2: 数据清洗及训练词向量
```
python pretrain_embedding.py
```
- raw_file： 文本语料
- clean_file： 清洗后语料
- seg_file： 分词后语料
- word2vec.txt：词向量文件

### Step 3: 训练模型
You can use the following command to train models for Sent-Track or Bag-Track:
```
python baseline.py --level sent 
python baseline.py --level bag
```
The model will be stored in `./model/` floder. We provide large scale unmarked corpus for train word vectors or language mdoels. The word vectors used in baseline system are trained by a package named gensim in python, and some parameters are set as follows:
```
from gensim.models import word2vec
model = word2vec.Word2Vec(sentences, sg=1, size=300, window=5, min_count=10, negative=5, sample=1e-4, workers=10)
```

### Step 4: 测试模型
You can use the following command to test models for Sent-Track or Bag-Track:
```
python baseline.py --mode test --level sent 
python baseline.py --mode test --level bag
```
Predicted results will be stored in result_sent.txt or result_bag.txt.


## 结果
我们的设备Intel 5118(12核) + 64GB RAM + RTX 2080Ti

模型 | 线下sent | 线上sent | 线下bag | 线上bag | 训练时长
---|---|---|---|---|---
官方（报告）| - | 0.22 | - | 0.31 | - | 4h40min(sent)
CNN with attention | 0.226087 | 0.21564 | | | 2h45min(sent)


## 参考文献
* Lin Y, Shen S, Liu Z, et al. Neural relation extraction with selective attention over instances[C]//Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2016, 1: 2124-2133.
