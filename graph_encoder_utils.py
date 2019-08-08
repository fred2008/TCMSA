import numpy as np
import tensorflow as tf
# import match_utils


def positional_encoding(max_len,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
        return shape: [max_len, num_units]
    '''

    T = max_len
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.range(T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])
        position_enc = position_enc.astype(np.float32)

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs


def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs


def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in xrange(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val


class GraphEncoder(object):
    def __init__(self, hidden_tensor, passage_nodes_size, is_training, config=None):

        assert config is not None
        #batch_size = config['train_batch_size']

        # placeholders

        # [batch_size, passage_nodes_size_max, passage_neighbors_size_max]
        self.passage_in_neighbor_indices = tf.placeholder(tf.int32, [None, None, None],
                                                          name='in_neighbor_indices')
        #self.passage_in_neighbor_edges = tf.placeholder(tf.int32, [None, None, None])
        self.passage_in_neighbor_mask = tf.placeholder(tf.float32, [None, None, None],
                                                       name='in_neighbor_mask')

        # [batch_size, passage_nodes_size_max, passage_neighbors_size_max]
        self.passage_out_neighbor_indices = tf.placeholder(tf.int32, [None, None, None],
                                                           name='out_neighbor_indices')
        #self.passage_out_neighbor_edges = tf.placeholder(tf.int32, [None, None, None])
        self.passage_out_neighbor_mask = tf.placeholder(tf.float32, [None, None, None],
                                                        name='out_neighbor_mask')

        self.passage_nodes_size = passage_nodes_size

        # shapes
        input_shape = tf.shape(self.passage_in_neighbor_indices)
        sent_num = batch_size = input_shape[0]
        sent_len = passage_nodes_size_max = input_shape[1]
        passage_in_neighbors_size_max = input_shape[2]
        passage_out_neighbors_size_max = tf.shape(self.passage_out_neighbor_indices)[2]

        # word representation for nodes, where each node only includes one word
        # [batch_size, passage_nodes_size_max, word_dim], [batch_size, passage_nodes_size_max]
        passage_node_representation, self.passage_nodes_mask = hidden_tensor

        print ( tf.expand_dims(self.passage_nodes_mask, axis=-1).dtype)
        print ( passage_node_representation.dtype)
        # apply the mask
        passage_node_representation = passage_node_representation * tf.expand_dims(self.passage_nodes_mask, axis=-1)

        input_dim = config['lstm_num_units']

        if config['compress_input']: # compress input word vector into smaller vectors
            # print([input_dim, config['compress_input_dim']])
            w_compress = tf.get_variable("w_compress_input", [input_dim, config['compress_input_dim']], dtype=tf.float32)
            b_compress = tf.get_variable("b_compress_input", [config['compress_input_dim']], dtype=tf.float32)

            passage_node_representation = tf.reshape(passage_node_representation, [-1, input_dim])
            passage_node_representation = tf.matmul(passage_node_representation, w_compress) + b_compress
            passage_node_representation = tf.tanh(passage_node_representation)
            passage_node_representation = tf.reshape(passage_node_representation,
                    [batch_size, passage_nodes_size_max, config['compress_input_dim']])
            input_dim = config['compress_input_dim']

        passage_node_representation = tf.cond(is_training,
                lambda: tf.nn.dropout(passage_node_representation, config['dropout_rate']),
                lambda: passage_node_representation)

        # ======Highway layer======
        if config['with_highway']:
            with tf.variable_scope("input_highway"):
                passage_node_representation = multi_highway_layer(passage_node_representation,
                        input_dim, config['highway_layer_num'])

        self.input_dim = input_dim

        with tf.variable_scope('graph_encoder'):
            # =========== in neighbor
            # [batch_size, passage_len, passage_neighbors_size_max, edge_dim]
            #passage_in_neighbor_edge_representations = tf.nn.embedding_lookup(self.edge_embedding,
            #        self.passage_in_neighbor_edges)
            # [batch_size, passage_len, passage_neighbors_size_max, input_dim]
            passage_in_neighbor_node_representations = self.collect_neighbor_node_representations(
                    passage_node_representation, self.passage_in_neighbor_indices)

            passage_in_neighbor_representations = tf.concat(
                    [passage_in_neighbor_node_representations], 3)
            passage_in_neighbor_representations = tf.multiply(passage_in_neighbor_representations,
                    tf.expand_dims(self.passage_in_neighbor_mask, axis=-1))
            # [batch_size, passage_len, input_dim + edge_dim]
            passage_in_neighbor_representations = tf.reduce_sum(passage_in_neighbor_representations, axis=2)

            # ============ out neighbor
            # [batch_size, passage_len, passage_neighbors_size_max, edge_dim]
            #passage_out_neighbor_edge_representations = tf.nn.embedding_lookup(self.edge_embedding,
            #        self.passage_out_neighbor_edges)
            # [batch_size, passage_len, passage_neighbors_size_max, input_dim]
            passage_out_neighbor_node_representations = self.collect_neighbor_node_representations(
                    passage_node_representation, self.passage_out_neighbor_indices)

            passage_out_neighbor_representations = tf.concat(
                    [passage_out_neighbor_node_representations], 3)
            passage_out_neighbor_representations = tf.multiply(passage_out_neighbor_representations,
                    tf.expand_dims(self.passage_out_neighbor_mask, axis=-1))
            # [batch_size, passage_len, input_dim + edge_dim]
            passage_out_neighbor_representations = tf.reduce_sum(passage_out_neighbor_representations, axis=2)

            if True:
                edge_dim = 0

            compress_vector_dim = input_dim+edge_dim
            # =====transpose neighbor_representations

            passage_in_neighbor_representations = tf.reshape(passage_in_neighbor_representations, 
                                                             [-1, input_dim+edge_dim])
            passage_out_neighbor_representations = tf.reshape(passage_out_neighbor_representations,
                                                              [-1, input_dim+edge_dim])


            # assume each node has a neighbor vector, and it is None at the beginning
            passage_node_hidden = tf.zeros([batch_size, passage_nodes_size_max, config['neighbor_vector_dim']])
            passage_node_cell = tf.zeros([batch_size, passage_nodes_size_max, config['neighbor_vector_dim']])

            w_in_ingate = tf.get_variable("w_in_ingate",
                    [compress_vector_dim, config['neighbor_vector_dim']], dtype=tf.float32)
            u_in_ingate = tf.get_variable("u_in_ingate",
                    [config['neighbor_vector_dim'], config['neighbor_vector_dim']], dtype=tf.float32)
            b_ingate = tf.get_variable("b_ingate",
                    [config['neighbor_vector_dim']], dtype=tf.float32)
            w_out_ingate = tf.get_variable("w_out_ingate",
                    [compress_vector_dim, config['neighbor_vector_dim']], dtype=tf.float32)
            u_out_ingate = tf.get_variable("u_out_ingate",
                    [config['neighbor_vector_dim'], config['neighbor_vector_dim']], dtype=tf.float32)

            w_in_forgetgate = tf.get_variable("w_in_forgetgate",
                    [compress_vector_dim, config['neighbor_vector_dim']], dtype=tf.float32)
            u_in_forgetgate = tf.get_variable("u_in_forgetgate",
                    [config['neighbor_vector_dim'], config['neighbor_vector_dim']], dtype=tf.float32)
            b_forgetgate = tf.get_variable("b_forgetgate",
                    [config['neighbor_vector_dim']], dtype=tf.float32)
            w_out_forgetgate = tf.get_variable("w_out_forgetgate",
                    [compress_vector_dim, config['neighbor_vector_dim']], dtype=tf.float32)
            u_out_forgetgate = tf.get_variable("u_out_forgetgate",
                    [config['neighbor_vector_dim'], config['neighbor_vector_dim']], dtype=tf.float32)

            w_in_outgate = tf.get_variable("w_in_outgate",
                    [compress_vector_dim, config['neighbor_vector_dim']], dtype=tf.float32)
            u_in_outgate = tf.get_variable("u_in_outgate",
                    [config['neighbor_vector_dim'], config['neighbor_vector_dim']], dtype=tf.float32)
            b_outgate = tf.get_variable("b_outgate",
                    [config['neighbor_vector_dim']], dtype=tf.float32)
            w_out_outgate = tf.get_variable("w_out_outgate",
                    [compress_vector_dim, config['neighbor_vector_dim']], dtype=tf.float32)
            u_out_outgate = tf.get_variable("u_out_outgate",
                    [config['neighbor_vector_dim'], config['neighbor_vector_dim']], dtype=tf.float32)

            w_in_cell = tf.get_variable("w_in_cell",
                    [compress_vector_dim, config['neighbor_vector_dim']], dtype=tf.float32)
            u_in_cell = tf.get_variable("u_in_cell",
                    [config['neighbor_vector_dim'], config['neighbor_vector_dim']], dtype=tf.float32)
            b_cell = tf.get_variable("b_cell",
                    [config['neighbor_vector_dim']], dtype=tf.float32)
            w_out_cell = tf.get_variable("w_out_cell",
                    [compress_vector_dim, config['neighbor_vector_dim']], dtype=tf.float32)
            u_out_cell = tf.get_variable("u_out_cell",
                    [config['neighbor_vector_dim'], config['neighbor_vector_dim']], dtype=tf.float32)

            # calculate question graph representation
            graph_representations = []
            n_step = config['num_syntax_match_layer']
            for i in range(config['num_syntax_match_layer']):
                # =============== in edge hidden
                # h_{ij} [batch_size, node_len, neighbors_size, neighbor_vector_dim]
                passage_in_edge_prev_hidden = self.collect_neighbor_node_representations(
                    passage_node_hidden, self.passage_in_neighbor_indices)
                passage_in_edge_prev_hidden = tf.multiply(passage_in_edge_prev_hidden,
                        tf.expand_dims(self.passage_in_neighbor_mask, axis=-1))
                # [batch_size, node_len, neighbor_vector_dim]
                passage_in_edge_prev_hidden = tf.reduce_sum(passage_in_edge_prev_hidden, axis=2)
                passage_in_edge_prev_hidden = tf.multiply(passage_in_edge_prev_hidden,
                        tf.expand_dims(self.passage_nodes_mask, axis=-1))
                passage_in_edge_prev_hidden = tf.reshape(passage_in_edge_prev_hidden,
                        [-1, config['neighbor_vector_dim']])

                # =============== out edge hidden
                # h_{jk} [batch_size, node_len, neighbors_size, neighbor_vector_dim]
                passage_out_edge_prev_hidden = self.collect_neighbor_node_representations(
                    passage_node_hidden, self.passage_out_neighbor_indices)
                passage_out_edge_prev_hidden = tf.multiply(passage_out_edge_prev_hidden,
                        tf.expand_dims(self.passage_out_neighbor_mask, axis=-1))
                # [batch_size, node_len, neighbor_vector_dim]
                passage_out_edge_prev_hidden = tf.reduce_sum(passage_out_edge_prev_hidden, axis=2)
                passage_out_edge_prev_hidden = tf.multiply(passage_out_edge_prev_hidden,
                        tf.expand_dims(self.passage_nodes_mask, axis=-1))
                passage_out_edge_prev_hidden = tf.reshape(passage_out_edge_prev_hidden,
                        [-1, config['neighbor_vector_dim']])

                ## ig
                passage_edge_ingate = tf.sigmoid(tf.matmul(passage_in_neighbor_representations, w_in_ingate)
                                          + tf.matmul(passage_in_edge_prev_hidden, u_in_ingate)
                                          + tf.matmul(passage_out_neighbor_representations, w_out_ingate)
                                          + tf.matmul(passage_out_edge_prev_hidden, u_out_ingate)
                                          + b_ingate)
                passage_edge_ingate = tf.reshape(passage_edge_ingate,
                        [batch_size, passage_nodes_size_max, config['neighbor_vector_dim']])
                ## fg
                passage_edge_forgetgate = tf.sigmoid(tf.matmul(passage_in_neighbor_representations, w_in_forgetgate)
                                          + tf.matmul(passage_in_edge_prev_hidden, u_in_forgetgate)
                                          + tf.matmul(passage_out_neighbor_representations, w_out_forgetgate)
                                          + tf.matmul(passage_out_edge_prev_hidden, u_out_forgetgate)
                                          + b_forgetgate)
                passage_edge_forgetgate = tf.reshape(passage_edge_forgetgate,
                        [batch_size, passage_nodes_size_max, config['neighbor_vector_dim']])
                ## og
                passage_edge_outgate = tf.sigmoid(tf.matmul(passage_in_neighbor_representations, w_in_outgate)
                                          + tf.matmul(passage_in_edge_prev_hidden, u_in_outgate)
                                          + tf.matmul(passage_out_neighbor_representations, w_out_outgate)
                                          + tf.matmul(passage_out_edge_prev_hidden, u_out_outgate)
                                          + b_outgate)
                passage_edge_outgate = tf.reshape(passage_edge_outgate,
                        [batch_size, passage_nodes_size_max, config['neighbor_vector_dim']])
                ## input
                passage_edge_cell_input = tf.tanh(tf.matmul(passage_in_neighbor_representations, w_in_cell)
                                          + tf.matmul(passage_in_edge_prev_hidden, u_in_cell)
                                          + tf.matmul(passage_out_neighbor_representations, w_out_cell)
                                          + tf.matmul(passage_out_edge_prev_hidden, u_out_cell)
                                          + b_cell)
                passage_edge_cell_input = tf.reshape(passage_edge_cell_input,
                        [batch_size, passage_nodes_size_max, config['neighbor_vector_dim']])

                passage_edge_cell = passage_edge_forgetgate * passage_node_cell + passage_edge_ingate * passage_edge_cell_input
                passage_edge_hidden = passage_edge_outgate * tf.tanh(passage_edge_cell)
                # node mask
                # [batch_size, passage_len, neighbor_vector_dim]
                passage_node_cell = tf.multiply(passage_edge_cell, tf.expand_dims(self.passage_nodes_mask, axis=-1))
                passage_node_hidden = tf.multiply(passage_edge_hidden, tf.expand_dims(self.passage_nodes_mask, axis=-1))

                graph_representations.append(passage_node_hidden)

            # decide how to use graph_representations
            self.graph_representations = graph_representations
            self.node_representations = passage_node_representation
            self.graph_hiddens = passage_node_hidden
            print('passage_node_hidden:', passage_node_hidden.get_shape().ndims)
            print('passage_node_hidden:', passage_node_hidden.get_shape() )
            self.graph_cells = passage_node_cell
            self.batch_size = batch_size

            # shape: [n_step, n_hidden]
            position_emb = positional_encoding(n_step,
                                               num_units=config['neighbor_vector_dim'],
                                               zero_pad=False,
                                               scale=False)

            # attention, shape:[sent_num, sent_len, n_hid]
            #query_att = passage_node_representation
            query_att = passage_node_hidden
            if config['att_vector_dim'] != compress_vector_dim:
                qry_w_att = tf.get_variable("qry_w_att", 
                                            [config['att_vector_dim'], compress_vector_dim],
                                            dtype=tf.float32)
                qry_b_att = tf.get_variable("qry_b_att", [compress_vector_dim], dtype=tf.float32)

                query_att = tf.nn.xw_plus_b(query_att, qry_w_att, qry_b_att)
                query_att = tf.tanh(query_att)

            query_att += position_emb[-1][None, None, :]

            # shape: [n_step, batch_size, sent_len, n_hidden]
            ml_hiddens = tf.stack(graph_representations, axis=0)

            ml_hiddens_position = ml_hiddens + position_emb[:, None, None, :]

            weight = tf.einsum('ijk,hijk->hij', query_att, ml_hiddens_position)
            weight = tf.nn.softmax(weight, dim=0)
            # shape: [batch_size, sent_len, n_hidden]

            weight_flt = tf.reshape(weight, [n_step, sent_num*sent_len])
            ml_hiddens_flt = tf.reshape(ml_hiddens, [n_step, sent_num*sent_len, compress_vector_dim])
            print("----------------")
            print(weight_flt.get_shape())
            print(ml_hiddens_flt.get_shape())
            hidden_att = tf.einsum('hi,hik->ik', weight_flt, ml_hiddens_flt)
            hidden_att = tf.reshape(hidden_att, [sent_num, sent_len, compress_vector_dim])
            self.hidden_att = hidden_att

    @staticmethod
    def collect_neighbor_node_representations(representation, positions):
        # representation: [batch_size, num_nodes, feature_dim]
        # positions: [batch_size, num_nodes, num_neighbors]
        feature_dim = tf.shape(representation)[2]
        input_shape = tf.shape(positions)
        batch_size = input_shape[0]
        num_nodes = input_shape[1]
        num_neighbors = input_shape[2]
        positions_flat = tf.reshape(positions, [batch_size, num_nodes*num_neighbors])
        def singel_instance(x):
            # x[0]: [num_nodes, feature_dim]
            # x[1]: [num_nodes*num_neighbors]
            return tf.gather(x[0], x[1])
        elems = (representation, positions_flat)
        representations = tf.map_fn(singel_instance, elems, dtype=tf.float32)
        return tf.reshape(representations, [batch_size, num_nodes, num_neighbors, feature_dim])

    @staticmethod
    def collect_final_step_lstm(lstm_rep, lens):
        lens = tf.maximum(lens, tf.zeros_like(lens, dtype=tf.int32)) # [batch,]
        idxs = tf.range(0, limit=tf.shape(lens)[0]) # [batch,]
        indices = tf.stack((idxs,lens,), axis=1) # [batch_size, 2]
        return tf.gather_nd(lstm_rep, indices, name='lstm-forward-last')
