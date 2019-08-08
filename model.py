import graph_encoder_utils
from tree_lstm import TreeLSTM
import tensorflow as tf
from util import load_namespace


class TreeCommunication:
    def __init__(self, config, weight_matrix, word_idx):

        self.config = config
        self.tree_lstm = tree_lstm = TreeLSTM(weight_matrix, word_idx, config)
        self.compiler = tree_lstm.compiler  # graph compiler for tree LSTM
        self.encoder = None

        # reloader for tree-LSTM
        # NOTICE: MUST reload before new Variable defined
        self.reloader = tf.train.Saver(tf.global_variables())

        # place holders
        self.keep_prob_ph = tree_lstm.keep_prob_ph
        self.emb_dropout = tree_lstm.emb_dropout
        self.is_train = tf.placeholder_with_default(False, [], name='is_train')
        self.labels = tf.placeholder(tf.int32, [None, None], name='sentiment_label')

    def forward(self):

        config = self.config
        tree_lstm = self.tree_lstm


        hiddens = tree_lstm.generate_hiddens()
        nodes_size = tree_lstm.nodes_size
        self.nodes_size = nodes_size

        output_layer = tree_lstm.output_layer
        max_l = tree_lstm.max_l

        mask = tf.sequence_mask(nodes_size, max_l, dtype=tf.float32)

        with tf.variable_scope('graph_lstm'):
            encoder = graph_encoder_utils.GraphEncoder(
                (hiddens, mask), nodes_size, self.is_train, self.config)
            self.encoder = encoder
            graph_hidden = encoder.graph_hiddens
            hidden_shape = tf.shape(graph_hidden)
            hidden_flt_shape = [hidden_shape[0] * hidden_shape[1], hidden_shape[-1]]
            logits_flt = tf.nn.xw_plus_b(tf.reshape(graph_hidden, hidden_flt_shape),
                                         output_layer._weights,
                                         output_layer._bias)
            # logits = tf.reshape(logits, [tf.shape(logits)[0]*tf.shape(logits)[1], tf.shape(logits)[-1]])

            labels_flt = tf.reshape(self.labels, [-1])
            mask_flt = tf.reshape(mask, [-1])
            loss_grp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_flt, labels=labels_flt)
            loss_grp = tf.reduce_sum(loss_grp * mask_flt)
            rglz_items = [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                          if 'bias' not in v.name and 'b_' not in v.name]

            loss_grp += tf.add_n(rglz_items) * 0.0005

            if config['all_vars_trained']:
                vars_graph = None
            else:
                vars_graph = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="graph_lstm")
                vars_graph += [output_layer._weights, output_layer._bias]

            metrics, metrics_num = self.eval_function(logits_flt, labels_flt,
                                                      mask_flt, nodes_size,
                                                      self.labels)

            return loss_grp, metrics, metrics_num, vars_graph

    def eval_function(self, logits_flt, labels_flt, mask_flt, nodes_size, labels):

        # all hits
        predictions_flt = tf.cast(tf.argmax(logits_flt, 1), tf.int32)
        hits = tf.cast(tf.equal(predictions_flt, labels_flt), tf.float32) * mask_flt

        # all binary hits
        softmax_flt = tf.nn.softmax(logits_flt)
        pred_flt_bin = (softmax_flt[:, 3] + softmax_flt[:, 4]) > (softmax_flt[:, 0] + softmax_flt[:, 1])
        label_flt_bin = labels_flt > 2
        label_flt_mask = tf.cast(tf.not_equal(labels_flt, 2), tf.float32) * mask_flt
        hits_bin_num = tf.reduce_sum(label_flt_mask)
        hits_bin = tf.cast(tf.equal(pred_flt_bin, label_flt_bin), tf.float32) * label_flt_mask

        # root hits
        last_idx = nodes_size - tf.ones_like(nodes_size)
        batch_size = tf.shape(labels)[0]
        max_len_sent = tf.shape(labels)[1]
        idxs = tf.range(batch_size) * max_len_sent + last_idx

        root_hits = tf.gather(hits, idxs)
        root_hits_bin = tf.gather(hits_bin, idxs)
        root_hits_bin_num = tf.reduce_sum(tf.gather(label_flt_mask, idxs))

        metrics = {"all_hits": hits, "all_binary_hits": hits_bin,
                   "root_hits": root_hits, "root_binary_hits": root_hits_bin}
        metrics = {k: tf.reduce_sum(v) for k, v in metrics.items()}
        sum_num = tf.reduce_sum(nodes_size)
        metrics_num = {"all_hits": sum_num, "all_binary_hits": hits_bin_num,
                       "root_hits": batch_size, "root_binary_hits": root_hits_bin_num}

        return metrics, metrics_num
