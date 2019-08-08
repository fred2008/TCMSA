import os
import sys
import itertools
from config import get_configs, update_config, get_config
from opt import Optimization
import datetime
from util import Timer, metric2str

import numpy as np
import tensorflow as tf
from data import Data_util, Dataset


def dev_eval(input_list, data_set, metrics, metrics_num, config, sess, vars_name):
    dev_metrics_total = {k: [] for k, v in metrics.items()}
    dev_metrics_num_total = {k: 0 for k, v in metrics_num.items()}

    for i, batch in enumerate(data_set.get_batches(None, config['batch_size_tr'])):

        dev_feed_dict = Dataset.gen_feed_dict(batch, input_list, vars_name, feed_dict=None)
        dev_rst = sess.run([metrics, metrics_num], dev_feed_dict)
        dev_metrics, dev_metrics_num = dev_rst

        for k, v in dev_metrics.items():
            dev_metrics_total[k].append(v)
            # print("  ", k, v, "num:",dev_metrics_num[k])

        for k, v in dev_metrics_num.items():
            dev_metrics_num_total[k] += v

    for k, v in dev_metrics_total.items():
        hits_total = np.sum(np.stack(v))
        dev_metrics[k] = hits_total / dev_metrics_num_total[k]

    return dev_metrics


if __name__ == "__main__":

    # 0. load config and session
    config = get_configs()

    if config['gpu_usage']:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config['gpu_usage'])
        print(config['gpu_usage'])
        gpu_config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(config=gpu_config)
    else:
        sess = tf.Session()

    save_path = [sys.argv[1]] if len(sys.argv) == 2 else []
    save_path.append(datetime.datetime.now().strftime('%H_%M_%S'))
    save_path = '_'.join(save_path)
    save_path = os.path.join("./model_saved", save_path)
    print("Save DIR:", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Step 1: prepare data
    data_dir = config['data_dir']
    print('saving files to %s' % data_dir)

    train_path, dev_path, test_path = Data_util.join(data_dir,
                                                     'trees/train.txt', 'trees/dev.txt', 'trees/test.txt')
    filtered_glove_path = os.path.join(data_dir, 'filtered_glove.txt')

    weight_matrix, word_idx = Data_util.load_embeddings(filtered_glove_path)

    tr_dataset = Dataset(train_path)
    dev_dataset = Dataset(dev_path)
    test_dataset = Dataset(test_path)
    n_train = len(tr_dataset.trees)

    from model import TreeCommunication
    tcm = TreeCommunication(config, weight_matrix, word_idx)
    loss_grp, metrics, metrics_num, vars_graph = tcm.forward()
    compiler = tcm.compiler

    train_grp, optmz = Optimization.apply_opt(config['lr_method'], loss_grp,
                                              var_list=vars_graph,
                                              clip=config['clip'])

    train_feed_dict = {tcm.keep_prob_ph: config['drop_out_keep'],
                       tcm.emb_dropout: config['emb_dropout'],
                       tcm.is_train: True}

    print(len(tf.global_variables()))
    for i, var in enumerate(tf.global_variables()):
        print("--------", i, var.name, var.get_shape())

    encoder = tcm.encoder
    input_vars = [compiler.loom_input_tensor, tcm.nodes_size, tcm.labels,
                  encoder.passage_in_neighbor_indices,
                  encoder.passage_in_neighbor_mask,
                  encoder.passage_out_neighbor_indices,
                  encoder.passage_out_neighbor_mask]

    vars_name = {name: i for i, name in enumerate(["n", "labels", "childs", "prts"])}

    tr_dataset.build_inputs(compiler)
    dev_dataset.build_inputs(compiler)
    test_dataset.build_inputs(compiler)

    best_dev_acc = 0.0
    best_test_acc = 0.0
    best_dev_mtrc = {k: 0. for k, _ in metrics.items()}

    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, config['model4test'])

    # evaluation
    dev_metrics = dev_eval(
        input_vars, dev_dataset, metrics, metrics_num, config, sess, vars_name
    )
    test_metrics = dev_eval(
        input_vars, test_dataset, metrics, metrics_num, config, sess, vars_name
    )

    dev_acc = dev_metrics['root_hits']
    test_acc = test_metrics['root_hits']

    print('dev: %s' % (' '.join(metric2str(dev_metrics))))
    print('test: %s' % (' '.join(metric2str(test_metrics))))
