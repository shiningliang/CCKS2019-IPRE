import pickle as pkl
import numpy as np
import os


def dump_pkl(file_path, obj):
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f)
    f.close()


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        obj = pkl.load(f)
    f.close()
    return obj


def pos_index(x, pos_limit):
    if x < -pos_limit:
        return 0
    if x >= -pos_limit and x <= pos_limit:
        return x + pos_limit + 1
    if x > pos_limit:
        return 2 * pos_limit + 2


def load_sent(filename, word_map, flags):
    sentence_dict = {}
    with open(filename, 'r') as fr:
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

            length = min(flags.sen_len, len(sentence))

            for i in range(length):
                words.append(word_map.get(sentence[i], word_map['UNK']))
                pos1.append(pos_index(i - en1_pos, flags.pos_limit))
                pos2.append(pos_index(i - en2_pos, flags.pos_limit))

            if length < flags.sen_len:
                for i in range(length, flags.sen_len):
                    words.append(word_map['PAD'])
                    pos1.append(pos_index(i - en1_pos, flags.pos_limit))
                    pos2.append(pos_index(i - en2_pos, flags.pos_limit))
            sentence_dict[id_] = np.reshape(np.asarray([words, pos1, pos2], dtype=np.int32), (1, 3, flags.sen_len))
        return sentence_dict


def create_wordVec(flags):
    word_map = {}
    word_map['PAD'] = len(word_map)
    word_map['UNK'] = len(word_map)
    word_embed = []
    for line in open(os.path.join(flags.raw_data_path, 'word2vec.txt')):
        content = line.strip().split()
        if len(content) != flags.word_dim + 1:
            continue
        word_map[content[0]] = len(word_map)
        word_embed.append(np.asarray(content[1:], dtype=np.float32))

    word_embed = np.stack(word_embed)
    embed_mean, embed_std = word_embed.mean(), word_embed.std()

    pad_embed = np.random.normal(embed_mean, embed_std, (2, flags.word_dim))
    word_embed = np.concatenate((pad_embed, word_embed), axis=0)
    word_embed = word_embed.astype(np.float32)
    print('Word in dict - {}'.format(len(word_map)))

    dump_pkl(os.path.join(flags.processed_data_path, 'word_map.pkl'), word_map)
    dump_pkl(os.path.join(flags.processed_data_path, 'word_embed.pkl'), word_embed)


def trans2ids(sentence_dict, level, relation_file, out_path, set_type, num_classes=35):
    if level == 'bag':
        all_bags = []
        all_sents = []
        all_labels = []
        with open(relation_file, 'r') as fr:
            for line in fr:
                rel = [0] * num_classes
                try:
                    bag_id, _, _, sents, types = line.strip().split('\t')
                    type_list = types.split()
                    for tp in type_list:
                        if len(type_list) > 1 and tp == '0':
                            # if a bag has multiple relations, we only consider non-NA relations
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
        out_path = os.path.join(out_path, set_type, level)
        dump_pkl(os.path.join(out_path, 'all_bags.pkl'), all_bags)
        dump_pkl(os.path.join(out_path, 'all_sents.pkl'), all_sents)
        dump_pkl(os.path.join(out_path, 'all_labels.pkl'), all_labels)
    else:
        all_sent_ids = []
        all_sents = []
        all_labels = []
        with open(relation_file, 'r') as fr:
            for line in fr:
                rel = [0] * num_classes
                try:
                    sent_id, types = line.strip().split('\t')
                    type_list = types.split()
                    for tp in type_list:
                        if len(type_list) > 1 and tp == '0':
                            # if a sentence has multiple relations, we only consider non-NA relations
                            continue
                        rel[int(tp)] = 1
                except:
                    sent_id = line.strip()

                all_sent_ids.append(sent_id)
                all_sents.append(sentence_dict[sent_id])

                all_labels.append(np.reshape(np.asarray(rel, dtype=np.float32), (-1, num_classes)))

        all_sents = np.concatenate(all_sents, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        out_path = os.path.join(out_path, set_type, level)
        dump_pkl(os.path.join(out_path, 'all_sent_ids.pkl'), all_sent_ids)
        dump_pkl(os.path.join(out_path, 'all_sents.pkl'), all_sents)
        dump_pkl(os.path.join(out_path, 'all_labels.pkl'), all_labels)


def create_serial(flags):
    levels = ['bag', 'sent']
    set_types = ['train', 'dev', 'test']
    with open(os.path.join(flags.processed_data_path, 'word_map.pkl'), 'rb') as fm:
        word_map = pkl.load(fm)

    for st in set_types:
        print('Transforming {} sets'.format(st))
        sent = load_sent(os.path.join(flags.raw_data_path, 'sent_' + st + '.txt'), word_map, flags)
        for l in levels:
            print('In level {}'.format(l))
            trans2ids(sent, l, os.path.join(flags.raw_data_path, l + '_relation_' + st + '.txt'),
                      flags.processed_data_path, st)
