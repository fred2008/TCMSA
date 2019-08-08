
import os
# configs


def get_configs():
    config = {}

    # GPU
    # config['gpu_usage'] = 0.53
    config['gpu_usage'] = 0.32

    # dataset
    config['data_dir'] = './data'
    config['model_dir'] = './model_saved'

    # network
    config['drop_out_keep'] = 0.75
    config['emb_dropout'] = 0.5
    # -- tree LSTM
    config['lstm_num_units'] = 300
    config['num_classes'] = 5

    # train
    # config['batch_size_tr'] = 100
    config['batch_size_tr'] = 24
    config['max_epochs'] = 20
    # config['max_epochs'] = 2
    config['pre_train'] = os.path.join(config['model_dir'], 'treelstm')
    config['clip'] = 1
    config['model4test'] = os.path.join(config['model_dir'], 'tcm_root_finegrain')
    config['eval_dev'] = 'root_hits'

    config['all_vars_trained'] = True
    # config['all_vars_trained'] = False
    # config['lr_method'] = 'sgd-lr_.005'
    config['lr_method'] = 'adagrad-lr_.05'
    config['emb_learning_factor'] = 0.1

    # eval
    config['batch_size_test'] = config['batch_size_tr']
    config['freq_eval_epoch'] = 10
    config['st_eval'] = 1

    # graph encoder
    config["isLower"] = True

    config["pointer_gen"] = True
    config["use_coverage"] = True
    config["attention_vec_size"] = 600
    config["batch_size"] = 100
    config["beam_size"] = 5

    config["num_syntax_match_layer"] = 9
    config["max_node_num"] = 180
    config["max_in_neigh_num"] = 2
    config["max_out_neigh_num"] = 8
    config["min_answer_len"] = 0
    config["max_answer_len"] = 50
    config["learning_rate"] = 0.0005
    config["lambda_l2"] = 1e-8
    config["dropout_rate"] = 0.8
    config["cov_loss_wt"] = 0.1
    config["max_epochs"] = 10
    config["optimize_type"] = "adam"

    config["with_highway"] = False
    config["highway_layer_num"] = 1

    config["with_char"] = False
    config["char_dim"] = 100
    config["char_lstm_dim"] = 100
    config["max_char_per_word"] = 20

    config["attention_type"] = "hidden"
    config["way_init_decoder"] = "all"
    config["edgelabel_dim"] = 100
    config["neighbor_vector_dim"] = 300
    config["fix_word_vec"] = False
    config["compress_input"] = False
    config["compress_input_dim"] = 300
    config["att_vector_dim"] = 300

    config["gen_hidden_size"] = 300
    config["num_softmax_samples"] = 100
    config["mode"] = "ce_train"

    return config


def update_config(epoch):

    config = {}
    with open('config.txt', 'r') as fin:
        lines = fin.readlines()

        for line in lines:
            name, ctt = line.rstrip().split(':')
            if not ctt:
                continue

            rules = ctt.split(';')
            for rule in rules:
                if not rule:
                    continue

                tok = rule.split(',')

                if len(tok) == 1:
                    config[name] = int(tok[0])

                elif len(tok) == 3:
                    start = int(tok[0])
                    end = int(tok[1])
                    value = int(tok[2])
                    if start <= epoch < end:
                        config[name] = value
                        break

                else:
                    raise ValueError('dynamic file input invalid')

    return config


def get_config(new_config, key, config):

    return new_config.get(key, config[key])

