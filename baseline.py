import numpy as np
import tensorflow as tf
import random
import os
import datetime
from collections import Counter


def set_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(2019)
    random.seed(2019)
    tf.set_random_seed(2019)


set_seed()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cuda', '0', 'gpu id')
tf.app.flags.DEFINE_boolean('pre_embed', True, 'load pre-trained word2vec')
tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size')
tf.app.flags.DEFINE_integer('epochs', 200, 'max train epochs')
tf.app.flags.DEFINE_integer('hidden_dim', 300, 'dimension of hidden embedding')
tf.app.flags.DEFINE_integer('word_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('pos_dim', 5, 'dimension of position embedding')
tf.app.flags.DEFINE_integer('pos_limit', 15, 'max distance of position embedding')
tf.app.flags.DEFINE_integer('sen_len', 60, 'sentence length')
tf.app.flags.DEFINE_integer('window', 3, 'window size')
tf.app.flags.DEFINE_string('model_path', './model', 'save model dir')
tf.app.flags.DEFINE_string('data_path', './data/raw', 'data dir to load')
tf.app.flags.DEFINE_string('level', 'bag', 'bag level or sentence level, option:bag/sent')
tf.app.flags.DEFINE_string('mode', 'train', 'train or test')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout rate')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('word_frequency', 5, 'minimum word frequency when constructing vocabulary list')


class Baseline:
    def __init__(self, flags):
        self.lr = flags.lr
        self.sen_len = flags.sen_len
        self.pre_embed = flags.pre_embed
        self.pos_limit = flags.pos_limit
        self.pos_dim = flags.pos_dim
        self.window = flags.window
        self.word_dim = flags.word_dim
        self.hidden_dim = flags.hidden_dim
        self.batch_size = flags.batch_size
        self.data_path = flags.data_path
        self.model_path = flags.model_path
        self.mode = flags.mode
        self.epochs = flags.epochs
        self.dropout = flags.dropout
        self.word_frequency = flags.word_frequency

        if flags.level == 'sent':
            self.bag = False
        elif flags.level == 'bag':
            self.bag = True
        else:
            self.bag = True

        self.pos_num = 2 * self.pos_limit + 3
        self.relation2id = self.load_relation()
        self.num_classes = len(self.relation2id)

        if self.pre_embed:
            self.wordMap, word_embed = self.load_wordVec()
            self.word_embedding = tf.get_variable(initializer=word_embed, name='word_embedding', trainable=False)

        else:
            self.wordMap = self.load_wordMap()
            self.word_embedding = tf.get_variable(shape=[len(self.wordMap), self.word_dim], name='word_embedding',
                                                  trainable=True)

        self.pos_e1_embedding = tf.get_variable(name='pos_e1_embedding', shape=[self.pos_num, self.pos_dim])
        self.pos_e2_embedding = tf.get_variable(name='pos_e2_embedding', shape=[self.pos_num, self.pos_dim])

        self.relation_embedding = tf.get_variable(name='relation_embedding', shape=[self.hidden_dim, self.num_classes])
        self.relation_embedding_b = tf.get_variable(name='relation_embedding_b', shape=[self.num_classes])

        self.sentence_reps = self.CNN_encoder()

        if self.bag:
            self.bag_level()
        else:
            self.sentence_level()
        self._classifier_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.classifier_loss)

    def pos_index(self, x):
        if x < -self.pos_limit:
            return 0
        if x >= -self.pos_limit and x <= self.pos_limit:
            return x + self.pos_limit + 1
        if x > self.pos_limit:
            return 2 * self.pos_limit + 2

    def load_wordVec(self):
        wordMap = {}
        wordMap['PAD'] = len(wordMap)
        wordMap['UNK'] = len(wordMap)
        word_embed = []
        for line in open(os.path.join(self.data_path, 'word2vec.txt')):
            content = line.strip().split()
            if len(content) != self.word_dim + 1:
                continue
            wordMap[content[0]] = len(wordMap)
            word_embed.append(np.asarray(content[1:], dtype=np.float32))

        word_embed = np.stack(word_embed)
        embed_mean, embed_std = word_embed.mean(), word_embed.std()

        pad_embed = np.random.normal(embed_mean, embed_std, (2, self.word_dim))
        word_embed = np.concatenate((pad_embed, word_embed), axis=0)
        word_embed = word_embed.astype(np.float32)
        return wordMap, word_embed

    def load_wordMap(self):
        wordMap = {}
        wordMap['PAD'] = len(wordMap)
        wordMap['UNK'] = len(wordMap)
        all_content = []
        for line in open(os.path.join(self.data_path, 'sent_train.txt')):
            all_content += line.strip().split('\t')[3].split()
        for item in Counter(all_content).most_common():
            if item[1] > self.word_frequency:
                wordMap[item[0]] = len(wordMap)
            else:
                break
        return wordMap

    def load_relation(self):
        relation2id = {}
        for line in open(os.path.join(self.data_path, 'relation2id.txt')):
            relation, id_ = line.strip().split()
            relation2id[relation] = int(id_)
        return relation2id

    def load_sent(self, filename):
        sentence_dict = {}
        with open(os.path.join(self.data_path, filename), 'r') as fr:
            for line in fr:
                id_, en1, en2, sentence = line.strip().split('\t')
                sentence = sentence.split()
                en1_pos = 0
                en2_pos = 0
                for i in range(len(sentence)):
                    if sentence[i] == en1:
                        en1_pos = i
                    if sentence[i] == en2:
                        en2_pos = i
                words = []
                pos1 = []
                pos2 = []

                length = min(self.sen_len, len(sentence))

                for i in range(length):
                    words.append(self.wordMap.get(sentence[i], self.wordMap['UNK']))
                    pos1.append(self.pos_index(i - en1_pos))
                    pos2.append(self.pos_index(i - en2_pos))

                if length < self.sen_len:
                    for i in range(length, self.sen_len):
                        words.append(self.wordMap['PAD'])
                        pos1.append(self.pos_index(i - en1_pos))
                        pos2.append(self.pos_index(i - en2_pos))
                sentence_dict[id_] = np.reshape(np.asarray([words, pos1, pos2], dtype=np.int32), (1, 3, self.sen_len))
        return sentence_dict

    def data_batcher(self, sentence_dict, filename, padding=False, shuffle=True):
        if self.bag:
            all_bags = []
            all_sents = []
            all_labels = []
            with open(os.path.join(self.data_path, filename), 'r') as fr:
                for line in fr:
                    rel = [0] * self.num_classes
                    try:
                        bag_id, _, _, sents, types = line.strip().split('\t')
                        type_list = types.split()
                        for tp in type_list:
                            if len(
                                    type_list) > 1 and tp == '0':  # if a bag has multiple relations, we only consider non-NA relations
                                continue
                            rel[int(tp)] = 1
                    except:
                        bag_id, _, _, sents = line.strip().split('\t')

                    sent_list = []
                    for sent in sents.split():
                        sent_list.append(sentence_dict[sent])

                    all_bags.append(bag_id)
                    all_sents.append(np.concatenate(sent_list, axis=0))
                    all_labels.append(np.asarray(rel, dtype=np.float32))

            self.data_size = len(all_bags)
            self.datas = all_bags
            data_order = list(range(self.data_size))
            if shuffle:
                np.random.shuffle(data_order)
            if padding:
                if self.data_size % self.batch_size != 0:
                    data_order += [data_order[-1]] * (self.batch_size - self.data_size % self.batch_size)

            for i in range(len(data_order) // self.batch_size):
                total_sens = 0
                out_sents = []
                out_sent_nums = []
                out_labels = []
                for k in data_order[i * self.batch_size:(i + 1) * self.batch_size]:
                    out_sents.append(all_sents[k])
                    out_sent_nums.append(total_sens)
                    total_sens += all_sents[k].shape[0]
                    out_labels.append(all_labels[k])

                out_sents = np.concatenate(out_sents, axis=0)
                out_sent_nums.append(total_sens)
                out_sent_nums = np.asarray(out_sent_nums, dtype=np.int32)
                out_labels = np.stack(out_labels)

                yield out_sents, out_labels, out_sent_nums
        else:
            all_sent_ids = []
            all_sents = []
            all_labels = []
            with open(os.path.join(self.data_path, filename), 'r') as fr:
                for line in fr:
                    rel = [0] * self.num_classes
                    try:
                        sent_id, types = line.strip().split('\t')
                        type_list = types.split()
                        for tp in type_list:
                            if len(
                                    type_list) > 1 and tp == '0':  # if a sentence has multiple relations, we only consider non-NA relations
                                continue
                            rel[int(tp)] = 1
                    except:
                        sent_id = line.strip()

                    all_sent_ids.append(sent_id)
                    all_sents.append(sentence_dict[sent_id])

                    all_labels.append(np.reshape(np.asarray(rel, dtype=np.float32), (-1, self.num_classes)))

            self.data_size = len(all_sent_ids)
            self.datas = all_sent_ids

            all_sents = np.concatenate(all_sents, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            data_order = list(range(self.data_size))
            if shuffle:
                np.random.shuffle(data_order)
            if padding:
                if self.data_size % self.batch_size != 0:
                    data_order += [data_order[-1]] * (self.batch_size - self.data_size % self.batch_size)

            for i in range(len(data_order) // self.batch_size):
                idx = data_order[i * self.batch_size:(i + 1) * self.batch_size]
                yield all_sents[idx], all_labels[idx], None

    def CNN_encoder(self):
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_word')
        self.input_pos_e1 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_pos_e1')
        self.input_pos_e2 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_pos_e2')
        self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='input_label')

        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(self.word_embedding, self.input_word), \
                                                   tf.nn.embedding_lookup(self.pos_e1_embedding, self.input_pos_e1), \
                                                   tf.nn.embedding_lookup(self.pos_e2_embedding, self.input_pos_e2)])
        inputs_forward = tf.expand_dims(inputs_forward, -1)

        with tf.name_scope('conv-maxpool'):
            w = tf.get_variable(name='w', shape=[self.window, self.word_dim + 2 * self.pos_dim, 1, self.hidden_dim])
            b = tf.get_variable(name='b', shape=[self.hidden_dim])
            conv = tf.nn.conv2d(
                inputs_forward,
                w,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv')
            h = tf.nn.bias_add(conv, b)
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sen_len - self.window + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool')
        sen_reps = tf.tanh(tf.reshape(pooled, [-1, self.hidden_dim]))
        sen_reps = tf.nn.dropout(sen_reps, self.keep_prob)
        return sen_reps

    def bag_level(self):
        self.classifier_loss = 0.0
        self.probability = []

        self.bag_sens = tf.placeholder(dtype=tf.int32, shape=[self.batch_size + 1], name='bag_sens')
        self.att_A = tf.get_variable(name='att_A', shape=[self.hidden_dim])
        self.rel = tf.reshape(tf.transpose(self.relation_embedding), [self.num_classes, self.hidden_dim])

        for i in range(self.batch_size):
            sen_reps = tf.reshape(self.sentence_reps[self.bag_sens[i]:self.bag_sens[i + 1]], [-1, self.hidden_dim])

            att_sen = tf.reshape(tf.multiply(sen_reps, self.att_A), [-1, self.hidden_dim])
            score = tf.matmul(self.rel, tf.transpose(att_sen))
            alpha = tf.nn.softmax(score, 1)
            bag_rep = tf.matmul(alpha, sen_reps)

            out = tf.matmul(bag_rep, self.relation_embedding) + self.relation_embedding_b

            prob = tf.reshape(tf.reduce_sum(tf.nn.softmax(out, 1) * tf.reshape(self.input_label[i], [-1, 1]), 0),
                              [self.num_classes])

            self.probability.append(
                tf.reshape(tf.reduce_sum(tf.nn.softmax(out, 1) * tf.diag([1.0] * (self.num_classes)), 1),
                           [-1, self.num_classes]))
            self.classifier_loss += tf.reduce_sum(
                -tf.log(tf.clip_by_value(prob, 1.0e-10, 1.0)) * tf.reshape(self.input_label[i], [-1]))

        self.probability = tf.concat(axis=0, values=self.probability)
        self.classifier_loss = self.classifier_loss / tf.cast(self.batch_size, tf.float32)

    def sentence_level(self):
        out = tf.matmul(self.sentence_reps, self.relation_embedding) + self.relation_embedding_b
        self.probability = tf.nn.softmax(out, 1)
        self.classifier_loss = tf.reduce_mean(
            tf.reduce_sum(-tf.log(tf.clip_by_value(self.probability, 1.0e-10, 1.0)) * self.input_label, 1))

    def run_train(self, sess, batch):

        sent_batch, label_batch, sen_num_batch = batch

        feed_dict = {}
        feed_dict[self.keep_prob] = self.dropout
        feed_dict[self.input_word] = sent_batch[:, 0, :]
        feed_dict[self.input_pos_e1] = sent_batch[:, 1, :]
        feed_dict[self.input_pos_e2] = sent_batch[:, 2, :]
        feed_dict[self.input_label] = label_batch
        if self.bag:
            feed_dict[self.bag_sens] = sen_num_batch

        _, classifier_loss = sess.run([self._classifier_train_op, self.classifier_loss], feed_dict)

        return classifier_loss

    def run_dev(self, sess, dev_batchers):
        all_labels = []
        all_probs = []
        for batch in dev_batchers:
            sent_batch, label_batch, sen_num_batch = batch
            all_labels.append(label_batch)

            feed_dict = {}
            feed_dict[self.keep_prob] = 1.0
            feed_dict[self.input_word] = sent_batch[:, 0, :]
            feed_dict[self.input_pos_e1] = sent_batch[:, 1, :]
            feed_dict[self.input_pos_e2] = sent_batch[:, 2, :]
            if self.bag:
                feed_dict[self.bag_sens] = sen_num_batch
            prob = sess.run([self.probability], feed_dict)
            all_probs.append(np.reshape(prob, (-1, self.num_classes)))

        all_labels = np.concatenate(all_labels, axis=0)[:self.data_size]
        all_probs = np.concatenate(all_probs, axis=0)[:self.data_size]
        if self.bag:
            all_preds = all_probs
            all_preds[all_probs > 0.9] = 1
            all_preds[all_probs <= 0.9] = 0
        else:
            all_preds = np.eye(self.num_classes)[np.reshape(np.argmax(all_probs, 1), (-1))]

        return all_preds, all_labels

    def run_test(self, sess, test_batchers):
        all_probs = []
        for batch in test_batchers:
            sent_batch, _, sen_num_batch = batch

            feed_dict = {}
            feed_dict[self.keep_prob] = 1.0
            feed_dict[self.input_word] = sent_batch[:, 0, :]
            feed_dict[self.input_pos_e1] = sent_batch[:, 1, :]
            feed_dict[self.input_pos_e2] = sent_batch[:, 2, :]
            if self.bag:
                feed_dict[self.bag_sens] = sen_num_batch
            prob = sess.run([self.probability], feed_dict)
            all_probs.append(np.reshape(prob, (-1, self.num_classes)))

        all_probs = np.concatenate(all_probs, axis=0)[:self.data_size]
        if self.bag:
            all_preds = all_probs
            all_preds[all_probs > 0.9] = 1
            all_preds[all_probs <= 0.9] = 0
        else:
            all_preds = np.eye(self.num_classes)[np.reshape(np.argmax(all_probs, 1), (-1))]

        if self.bag:
            with open('result_bag.txt', 'w') as fw:
                for i in range(self.data_size):
                    rel_one_hot = [int(num) for num in all_preds[i].tolist()]
                    rel_list = []
                    for j in range(0, self.num_classes):
                        if rel_one_hot[j] == 1:
                            rel_list.append(str(j))
                    if len(rel_list) == 0:  # if a bag has no relation, it will be consider as having a relation NA
                        rel_list.append('0')
                    fw.write(self.datas[i] + '\t' + ' '.join(rel_list) + '\n')
        else:
            with open('result_sent.txt', 'w') as fw:
                for i in range(self.data_size):
                    rel_one_hot = [int(num) for num in all_preds[i].tolist()]
                    rel_list = []
                    for j in range(0, self.num_classes):
                        if rel_one_hot[j] == 1:
                            rel_list.append(str(j))
                    fw.write(self.datas[i] + '\t' + ' '.join(rel_list) + '\n')

    def run_model(self, sess, saver):
        if self.mode == 'train':
            global_step = 0
            sent_train = self.load_sent('sent_train.txt')
            sent_dev = self.load_sent('sent_dev.txt')
            max_f1 = 0.0

            if not os.path.isdir(self.model_path):
                os.mkdir(self.model_path)

            for epoch in range(self.epochs):
                if self.bag:
                    train_batchers = self.data_batcher(sent_train, 'bag_relation_train.txt', padding=False,
                                                       shuffle=True)
                else:
                    train_batchers = self.data_batcher(sent_train, 'sent_relation_train.txt', padding=False,
                                                       shuffle=True)
                for batch in train_batchers:

                    losses = self.run_train(sess, batch)
                    global_step += 1
                    if global_step % 50 == 0:
                        time_str = datetime.datetime.now().isoformat()
                        tempstr = "{}: step {}, classifier_loss {:g}".format(time_str, global_step, losses)
                        print(tempstr)
                    if global_step % 200 == 0:
                        if self.bag:
                            dev_batchers = self.data_batcher(sent_dev, 'bag_relation_dev.txt', padding=True,
                                                             shuffle=False)
                        else:
                            dev_batchers = self.data_batcher(sent_dev, 'sent_relation_dev.txt', padding=True,
                                                             shuffle=False)
                        all_preds, all_labels = self.run_dev(sess, dev_batchers)

                        # when calculate f1 score, we don't consider whether NA results are predicted or not
                        # the number of non-NA answers in test is counted as n_std
                        # the number of non-NA answers in predicted answers is counted as n_sys
                        # intersection of two answers is counted as n_r
                        n_r = int(np.sum(all_preds[:, 1:] * all_labels[:, 1:]))
                        n_std = int(np.sum(all_labels[:, 1:]))
                        n_sys = int(np.sum(all_preds[:, 1:]))
                        try:
                            precision = n_r / n_sys
                            recall = n_r / n_std
                            f1 = 2 * precision * recall / (precision + recall)
                        except ZeroDivisionError:
                            f1 = 0.0

                        if f1 > max_f1:
                            max_f1 = f1
                            print('f1: %f' % f1)
                            print('saving model')
                            path = saver.save(sess, os.path.join(self.model_path, 'ipre_bag_%d' % (self.bag)),
                                              global_step=0)
                            tempstr = 'have saved model to ' + path
                            print(tempstr)

        else:
            path = os.path.join(self.model_path, 'ipre_bag_%d' % self.bag) + '-0'
            tempstr = 'load model: ' + path
            print(tempstr)
            try:
                saver.restore(sess, path)
            except:
                raise ValueError('Unvalid model name')

            sent_test = self.load_sent('sent_test.txt')
            if self.bag:
                test_batchers = self.data_batcher(sent_test, 'bag_relation_test.txt', padding=True, shuffle=False)
            else:
                test_batchers = self.data_batcher(sent_test, 'sent_relation_test.txt', padding=True, shuffle=False)

            self.run_test(sess, test_batchers)


def main(_):
    tf.reset_default_graph()
    print('build model')
    gpu_options = tf.GPUOptions(visible_device_list=FLAGS.cuda, allow_growth=True)
    with tf.Graph().as_default():
        set_seed()
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1))
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('', initializer=initializer):
                model = Baseline(FLAGS)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
            model.run_model(sess, saver)


if __name__ == '__main__':
    tf.app.run()
