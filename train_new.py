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
        # gpu_options=tf.GPUOptions(allow_growth=True)
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

    full_glove_path, = Data_util.join(data_dir, 'glove.840B.300d.txt')
    train_path, dev_path, test_path = Data_util.join(data_dir,
                                                     'trees/train.txt', 'trees/dev.txt', 'trees/test.txt')

    filtered_glove_path = os.path.join(data_dir, 'filtered_glove.txt')
    if not os.path.isfile(filtered_glove_path):
        Data_util.filter_glovefilter(data_dir, full_glove_path, filtered_glove_path)

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

    sess.run(tf.global_variables_initializer())

    if config['pre_train']:
        tcm.reloader.restore(sess, config['pre_train'])
        print("Loading Model from", config['pre_train'])

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

    import tensorflow_fold as td

    best_dev_acc = 0.0

    saver = tf.train.Saver(max_to_keep=None)
    save_prfx = os.path.join(save_path, 'graph_lstm')

    ceildiv = lambda a, b: -(-a // b)
    n_batchs = ceildiv(n_train, config['batch_size_tr'])
    # interval_eval = ceildiv(n_batchs, config['freq_eval_epoch'])

    # print("Time: process, build_model", timer.spot())

    timer2 = Timer()
    for epoch, shuffled in enumerate(tr_dataset.get_shuffled_epoches()):

        new_config = update_config(epoch)
        max_epochs = get_config(new_config, 'max_epochs', config)
        if epoch >= max_epochs:
            break

        interval_eval = ceildiv(n_batchs, get_config(new_config, 'freq_eval_epoch', config))
        # interval_eval = ceildiv(n_batchs, config['freq_eval_epoch'])

        # train per epoch
        losses = []
        for i, batch in enumerate(td.group_by_batches(shuffled, config['batch_size_tr'])):
            feed_dict_batch = Dataset.gen_feed_dict(batch, input_vars, vars_name, train_feed_dict)
            _, batch_loss = sess.run([train_grp, loss_grp], feed_dict_batch)

            losses.append(batch_loss)

            if config['st_eval'] and epoch < config['st_eval']:
                continue

            if (i + 1) % interval_eval != 0 and i != n_batchs - 1:
                continue

            # print("Time: train n batch", timer.spot())

            train_loss = np.array(losses).mean()
            losses = []
            print('--------  epoch/batch: %d/%d, loss: %.3e ------------'
                  % (epoch, i, train_loss))

            # evalation
            dev_metrics = dev_eval(
                input_vars, dev_dataset, metrics, metrics_num, config, sess, vars_name
            )
            test_metrics = dev_eval(
                input_vars, test_dataset, metrics, metrics_num, config, sess, vars_name
            )

            dev_acc = dev_metrics[config['eval_dev']]

            print('epc: %d/%d, dev: %s' % (epoch, i, ' '.join(metric2str(dev_metrics))))
            print('epc: %d/%d, test: %s' % (epoch, i, ' '.join(metric2str(test_metrics))))

            # decide when to save
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc

                save_file_name = '%s_%d_%d' % (save_prfx, epoch, i)
                checkpoint_path = saver.save(sess, save_file_name)
                print('model saved in file: %s' % checkpoint_path)

        print("Time2: 1 epoch", timer2.spot())
