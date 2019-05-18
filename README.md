# A Baseline System For CCKS-2019-IPRE

## Introduction
We provide a baseline system based on convolutional neural network with selective attention.

## Getting Started
### Environment Requirements
* python 3.6
* numpy
* tensorflow 1.12.0
### Step 1: Download data
Please download the data from [the competition website](https://biendata.com/competition/ccks_2019_ipre/data/), then unzip files and put them in `./data/` folder.

### Step 2: Train the model
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
### Step 3: Test the model
You can use the following command to test models for Sent-Track or Bag-Track:
```
python baseline.py --mode test --level sent 
python baseline.py --mode test --level bag
```
Predicted results will be stored in result_sent.txt or result_bag.txt.
## Evaluation
We use f1 score as the basic evaluation metric to measure the performance of systems. In our baseline system, we get about 0.22 f1 score in Sent-track and about 0.31 f1 score in Bag-Track by using pre-trained word vectors.
## References
* Lin Y, Shen S, Liu Z, et al. Neural relation extraction with selective attention over instances[C]//Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2016, 1: 2124-2133.
