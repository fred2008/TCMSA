import os, sys
import codecs
import numpy as np
from nltk.tokenize import sexpr
import itertools
import json


class Data_util:

    @staticmethod
    def join(data_dir, *file_names):
        out_paths = list(map(lambda x: os.path.join(data_dir, x), file_names))
        return out_paths

    @staticmethod
    def load_embeddings(embedding_path):
        """
        Loads embedings, returns weight matrix and dict from words to indices.
        """
        print('loading word embeddings from %s' % embedding_path)
        weight_vectors = []
        word_idx = {}
        with codecs.open(embedding_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(u' ', 1)
                word_idx[word] = len(weight_vectors)
                weight_vectors.append(np.array(vec.split(), dtype=np.float32))
        # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
        # '-RRB-' respectively in the parse-trees.
        word_idx[u'-LRB-'] = word_idx.pop(u'(')
        word_idx[u'-RRB-'] = word_idx.pop(u')')
        # Random embedding vector for unknown words.
        weight_vectors.append(np.random.uniform(
            -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
        return np.stack(weight_vectors), word_idx

    @staticmethod
    def filter_glovefilter(data_dir, full_glove_path, filtered_glove_path):
        vocab = set()
        # Download the full set of unlabeled sentences separated by '|'.
        sentence_path, = download_and_unzip(
            'http://nlp.stanford.edu/~socherr/', 'stanfordSentimentTreebank.zip', data_dir,
            'stanfordSentimentTreebank/SOStr.txt')
        with codecs.open(sentence_path, encoding='utf-8') as f:
            for line in f:
                # Drop the trailing newline and strip backslashes. Split into words.
                vocab.update(line.strip().replace('\\', '').split('|'))
        nread = 0
        nwrote = 0
        with codecs.open(full_glove_path, encoding='utf-8') as f:
            with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
                for line in f:
                    nread += 1
                    line = line.strip()
                    if not line: continue
                    if line.split(u' ', 1)[0] in vocab:
                        out.write(line + '\n')
                        nwrote += 1
        print('read %s lines, wrote %s' % (nread, nwrote))


class Dataset:

    def __init__(self, path):

        self.trees = self.load_trees(path)  # list of string

    def load_trees(self, filename):

        with codecs.open(filename, encoding='utf-8') as f:
            # Drop the trailing newline and strip \s.
            trees = [line.strip().replace('\\', '') for line in f]
            print('loaded %s trees from %s' % (len(trees), filename))
            return trees

    def build_inputs(self, compiler):
        # build feed_dict value
        data_set = compiler.build_loom_inputs(self.trees)
        data_set_info = map(self.get_feed_values, self.trees)
        self.data_set = zip(data_set, data_set_info)

    def get_shuffled_epoches(self, num_epoch=None):
        """
        return a list of shuffled epoches
        """
        import tensorflow_fold as td
        return td.epochs(self.data_set, num_epoch)

    def get_dataset(self):

        self.data_set, dataset = itertools.tee(self.data_set)
        return dataset

    def get_batches(self, dataset=None, batch_size=100):

        if dataset == None:
            dataset = self.get_dataset()
        import tensorflow_fold as td
        return td.group_by_batches(dataset, batch_size)

    @staticmethod
    def get_feed_values(sent):
        index = 0
        labels = []
        childs = []
        prts = []
        leaf_cnt = 0

        def post_order(s):
            nonlocal index
            nonlocal leaf_cnt
            label, phrase = s[1:-1].split(None, 1)
            leafs = sexpr.sexpr_tokenize(phrase)

            if len(leafs) == 2:
                lstr, rstr = leafs
                lrst = post_order(lstr)
                rrst = post_order(rstr)
                prts[lrst].append(index)
                prts[rrst].append(index)
            else:
                leaf_cnt += 1

            labels.append(label)
            prts.append([])
            # childs.append([lrst, rrst] if len(leafs)==2 else [])
            cur = index
            childs_list = [lrst, rrst] if len(leafs) == 2 else []
            childs_list.append(cur)  # self-loop
            childs.append(childs_list)
            # print(index)
            index += 1

            return cur

        last = post_order(sent)
        n = len(labels)
        assert last + 1 == n

        return n, labels, childs, prts, leaf_cnt

    @staticmethod
    def gen_feed_dict(batch, input_list, vars_name, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}

        keys = []  # store the key values
        loom_input = list(map(lambda x: x[0], batch))
        keys.append(loom_input)

        info = list(map(lambda x: x[1], batch))

        n_sent = list(map(lambda x: x[vars_name["n"]], info))
        max_node_len = max(n_sent)
        batch_size = len(n_sent)
        keys.append(n_sent)

        labels = np.zeros([batch_size, max_node_len], np.int32)
        childs, prts = [], []
        for i, s in enumerate(info):
            label = s[vars_name['labels']]
            labels[i][:len(label)] = label

            childs.append(s[vars_name['childs']])
            prts.append(s[vars_name['prts']])

        keys.append(labels)

        def pad_3d(indices):
            # neigh_indices is a 3D nested list
            # shape: [batch_size, max_len, max_adj]
            indices_flat = [char for word in indices for char in word]
            max_adj = max(list(map(len, indices_flat)))
            out_shape = [batch_size, max_node_len, max_adj]

            indices_padded = np.zeros(out_shape, np.int32)
            mask = np.zeros(out_shape, np.int32)

            for i in range(batch_size):
                for j in range(len(indices[i])):
                    adj_len = len(indices[i][j])
                    indices_padded[i][j][:adj_len] = indices[i][j]
                    mask[i][j][:adj_len] = 1

            return indices_padded, mask

        # print(childs)
        # print(prts)
        in_neigh_indices, in_neigh_mask = pad_3d(childs)
        out_neigh_indices, out_neigh_mask = pad_3d(prts)

        keys.append(in_neigh_indices)
        keys.append(in_neigh_mask)
        keys.append(out_neigh_indices)
        keys.append(out_neigh_mask)

        assert len(input_list) == len(keys)

        feed_dict = {k: v for k, v in list(zip(input_list, keys)) + list(feed_dict.items())}

        return feed_dict

    def gen_feed_dict2(batch, input_list, vars_name, feed_dict=None):
        if feed_dict == None:
            feed_dict = {}

        keys = []  # store the key values
        loom_input = list(map(lambda x: x[0], batch))
        keys.append(loom_input)

        info = list(map(lambda x: x[1], batch))

        n_sent = list(map(lambda x: x[vars_name["n"]], info))
        max_node_len = max(n_sent)
        batch_size = len(n_sent)
        keys.append(n_sent)

        labels = np.zeros([batch_size, max_node_len], np.int32)
        childs, prts = [], []
        leaf_cnt = []
        for i, s in enumerate(info):
            label = s[vars_name['labels']]
            labels[i][:len(label)] = label

            childs.append(s[vars_name['childs']])
            prts.append(s[vars_name['prts']])
            leaf_cnt.append(s[-1])

        keys.append(labels)

        def pad_3d(indices):
            # neigh_indices is a 3D nested list
            # shape: [batch_size, max_len, max_adj]
            indices_flat = [char for word in indices for char in word]
            max_adj = max(list(map(len, indices_flat)))
            out_shape = [batch_size, max_node_len, max_adj]

            indices_padded = np.zeros(out_shape, np.int32)
            mask = np.zeros(out_shape, np.int32)

            for i in range(batch_size):
                for j in range(len(indices[i])):
                    adj_len = len(indices[i][j])
                    indices_padded[i][j][:adj_len] = indices[i][j]
                    mask[i][j][:adj_len] = 1

            return indices_padded, mask

        # print(childs)
        # print(prts)
        in_neigh_indices, in_neigh_mask = pad_3d(childs)
        out_neigh_indices, out_neigh_mask = pad_3d(prts)

        keys.append(in_neigh_indices)
        keys.append(in_neigh_mask)
        keys.append(out_neigh_indices)
        keys.append(out_neigh_mask)

        assert len(input_list) == len(keys)

        feed_dict = {k: v for k, v in list(zip(input_list, keys)) + list(feed_dict.items())}

        return feed_dict, leaf_cnt

    @staticmethod
    def extract_word(sent):
        words = []
        labels = []

        def post_order(s):
            label, phrase = s[1:-1].split(None, 1)
            leafs = sexpr.sexpr_tokenize(phrase)

            if len(leafs) == 2:
                lstr, rstr = leafs
                post_order(lstr)
                post_order(rstr)
            else:
                words.append(leafs[0])
                labels.append(label)

            return label

        root = post_order(sent)
        # print(' '.join(words) + '\001' + ' '.join([i+'-'+j for i, j in zip(words, labels)]))
        # print(' '.join([i.lower()+'/'+j for i, j in zip(words, labels)]))
        print(' '.join(words))

        # for i, j in zip(words, labels):
        #     print(i.lower(), j)


def download_and_unzip(url_base, zip_name, data_dir, *file_names):

    out_paths = list(map(lambda x: os.path.join(data_dir, x), file_names))
    return out_paths


