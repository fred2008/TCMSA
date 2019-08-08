import tensorflow as tf
import tensorflow_fold as td
from nltk.tokenize import sexpr

import numpy as np
import tensorflow_fold.blocks.blocks
import tensorflow_fold.blocks.result_types as tdt


class Embedding_drop(td.TensorToTensorLayer):
  """An embedding for integers.

  Embeddings require integer scalars as input, and build `float32` vector
  outputs. Embeddings can be applied to multiple inputs. `Embedding` doesn't
  do any hashing on its own, it just takes its inputs mod `num_buckets`
  to determine which embedding(s) to return.

  Implementation detail: `tf.gather` currently only supports `int32`
  and `int64`. If the input type is smaller than 32 bits it will be
  cast to `tf.int32`. Since all currently defined TF dtypes other than
  `int32` and `int64` have less than 32 bits, this means that we
  support all current integer dtypes.
  """

  def __init__(self, num_buckets, num_units_out, initializer=None, name=None,
               trainable=True, mod_inputs=True, dropout_rate=None):
    """Initializes the layer.

    Args:
      num_buckets: How many buckets the embedding has.
      num_units_out: The number of output units in the layer.
      initializer: the initializer for the weights. Defaults to uniform unit
        scaling. The initializer can also be a Tensor or numpy array, in which
        case the weights are initialized to this value and shape. Note that in
        this case the weights will still be trainable unless you also pass
        `trainable=False`.
      name: An optional string name. Defaults to
        `Embedding_%d_%d % (num_buckets, num_units_out)`. Used to name the
        variable scope where the variables for the layer live.
      trainable: Whether or not to make the weights trainable.
      mod_inputs: Whether or not to mod the input by the number of buckets.

    Raises:
      ValueError: If the shape of `weights` is not
        `(num_buckets, num_units_out)`.
    """
    self._weights_shape = (num_buckets, num_units_out)
    if name is None: name = 'Embedding_%d_%d' % self._weights_shape
    if initializer is None:
      initializer = tf.uniform_unit_scaling_initializer(1.0)
    elif isinstance(initializer, np.ndarray):
      initializer = tf.convert_to_tensor(initializer)
    if isinstance(initializer, tf.Tensor):
      initializer.set_shape(self._weights_shape)
      self._weights_shape = None  # otherwise get_variable barfs
    self._initializer = initializer
    self._num_buckets = num_buckets
    self._num_units_out = num_units_out
    self._trainable = trainable
    self._mod_inputs = bool(mod_inputs)
    super(Embedding_drop, self).__init__(
        output_type=tdt.TensorType([num_units_out]), name_or_scope=name)
    self.dropout_rate = dropout_rate

  def _create_variables(self):
    if self.input_type.ndim != 0:
      raise TypeError('Embeddings take scalar inputs.')
    dtype = tf.as_dtype(self.input_type.dtype)
    if not dtype.is_integer: raise TypeError('Embeddings take integer inputs.')
    if dtype not in (tf.int32, tf.int64):  # only dtypes supported by tf.gather
      if np.iinfo(dtype.as_numpy_dtype).max > 2147483647:
         # pedantic future-proofing to handle hypothetical tf.uint64
        raise TypeError('cannot gather or upcast dtype %s' % dtype)
      self._cast = True
    else:
      self._cast = False
    self._weights = tf.get_variable(
        'weights', self._weights_shape, initializer=self._initializer,
        trainable=self._trainable)
    if self.dropout_rate != None:
      self.dropout_rate = tf.nn.dropout(self._weights, self.dropout_rate)


  @property
  def weights(self):
    if not self._created_variables:
      raise RuntimeError('weights have not been created; call the layer first')
    return self._weights

  @property
  def num_buckets(self):
    return self._num_buckets

  @property
  def num_units_out(self):
    return self._num_units_out

  def _process_batch(self, batch):
    # We have to call tf.abs before calling tf.mod, because tf.mod gives
    # native outputs when given negative inputs.
    if self._cast: batch = tf.cast(batch, tf.int32)
    if self._mod_inputs: batch = tf.mod(tf.abs(batch), self._num_buckets)
    return tf.gather(self._weights, batch)


class BinaryTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    """LSTM with two state inputs.

    This is the model described in section 3.2 of 'Improved Semantic
    Representations From Tree-Structured Long Short-Term Memory
    Networks' <http://arxiv.org/pdf/1503.00075.pdf>, with recurrent
    dropout as described in 'Recurrent Dropout without Memory Loss'
    <http://arxiv.org/pdf/1603.05118.pdf>.
    """

    def __init__(self, num_units, keep_prob=1.0):
        """Initialize the cell.

        Args:
            num_units: int, The number of units in the LSTM cell.
            keep_prob: Keep probability for recurrent dropout.
        """
        super(BinaryTreeLSTMCell, self).__init__(num_units)
        self._keep_prob = keep_prob

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            lhs, rhs = state
            c0, h0 = lhs
            c1, h1 = rhs
            concat = tf.contrib.layers.linear(
                tf.concat([inputs, h0, h1], 1), 5 * self._num_units)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f0, f1, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            j = self._activation(j)
            if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
                j = tf.nn.dropout(j, self._keep_prob)

            new_c = (c0 * tf.sigmoid(f0 + self._forget_bias) +
                     c1 * tf.sigmoid(f1 + self._forget_bias) +
                     tf.sigmoid(i) * j)
            new_h = self._activation(new_c) * tf.sigmoid(o)

            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


class TreeLSTM:
    def __init__(self, weight_matrix, word_idx, config):
        """
        return compiler
        """
        self.keep_prob_ph = tf.placeholder_with_default(1.0, [], name='keep_prob_ph')
        self.emb_dropout = tf.placeholder_with_default(1.0, [], name='emb_dropout')

        self.compiler = self.build_network(weight_matrix, word_idx, config)

    def build_network(self, weight_matrix, word_idx, config):

        # Tai et al. used 150, but our regularization strategy is more effective
        keep_prob_ph = self.keep_prob_ph
        emb_dropout = self.emb_dropout
        lstm_num_units = config['lstm_num_units']
        self.tree_lstm = td.ScopedLayer(
            tf.contrib.rnn.DropoutWrapper(
                BinaryTreeLSTMCell(lstm_num_units, keep_prob=keep_prob_ph),
                input_keep_prob=keep_prob_ph,
                output_keep_prob=keep_prob_ph),
            name_or_scope='tree_lstm')

        NUM_CLASSES = config['num_classes']  # number of distinct sentiment labels

        self.output_layer = td.FC(NUM_CLASSES, activation=None, name='output_layer')
        # self.word_embedding = td.Embedding(*weight_matrix.shape,
        #                                    initializer=weight_matrix,
        #                                    name='word_embedding',
        #                                    dropout_rate=emb_dropout)
        self.word_embedding = Embedding_drop(*weight_matrix.shape,
                                           initializer=weight_matrix,
                                           name='word_embedding',
                                           dropout_rate=emb_dropout)

        self.embed_subtree = td.ForwardDeclaration(name='embed_subtree')

        model = self.embed_tree(self.logits_and_state(word_idx),
                                is_root=True)  # root

        self.embed_subtree.resolve_to(self.embed_tree(
            self.logits_and_state(word_idx),
            is_root=False))

        compiler = td.Compiler.create(model)
        # print('input type: %s' % model.input_type)
        # print('output type: %s' % model.output_type)

        return compiler

    def generate_hiddens(self):

        # nodes_size = tf.placeholder(tf.int32, [None])
        self.nodes_size = nodes_size = tf.placeholder(tf.int32, [None])

        self.max_l = max_l = tf.reduce_max(nodes_size)

        hidden_flt = self.compiler.metric_tensors['hidden']
        # hiddens = tf.split(hidden_flt, nodes_size) # split cannot dynamic length
        split_end = tf.cumsum(nodes_size)
        split_begin = tf.concat([tf.zeros([1], split_end.dtype), split_end[:-1]], axis=0)

        def fn_extract_pad(a, x):
            paddings = [[0, max_l - (x[1] - x[0])], [0, 0]]
            return tf.pad(hidden_flt[x[0]:x[1]], paddings)

        output_init = tf.zeros([max_l, tf.shape(hidden_flt)[-1]], dtype=tf.float32)

        # from flt to paded represent, shape: batch_size, max_l, num_hidden
        hiddens = tf.scan(fn=fn_extract_pad,
                          elems=(split_begin, split_end),
                          initializer=output_init)
        # hiddens = [tf.pad(hidden_sent, [[0, max_l]]) for hidden_sent in hiddens]
        # hiddens = tf.stack(hiddens) # shape: batch_size, max_l, num_hidden

        return hiddens

    @staticmethod
    def add_metrics(is_root, is_neutral):
        """A block that adds metrics for loss and hits; output is the LSTM state."""
        c = td.Composition(
            name='predict(is_root=%s, is_neutral=%s)' % (is_root, is_neutral))
        with c.scope():
            # destructure the input; (labels, (logits, state))
            labels = c.input[0]
            logits = td.GetItem(0).reads(c.input[1])
            state = td.GetItem(1).reads(c.input[1])

            # calculate loss
            loss = td.Function(TreeLSTM.tf_node_loss)
            td.Metric('all_loss').reads(loss.reads(logits, labels))
            if is_root: td.Metric('root_loss').reads(loss)

            # calculate fine-grained hits
            hits = td.Function(TreeLSTM.tf_fine_grained_hits)
            td.Metric('all_hits').reads(hits.reads(logits, labels))
            if is_root: td.Metric('root_hits').reads(hits)

            # calculate binary hits, if the label is not neutral
            if not is_neutral:
                binary_hits = td.Function(TreeLSTM.tf_binary_hits).reads(logits, labels)
                td.Metric('all_binary_hits').reads(binary_hits)
                if is_root: td.Metric('root_binary_hits').reads(binary_hits)

            # add hidden of each node as Metric
            td.Metric('hidden').reads(td.GetItem(1).reads(state))

            # output the state, which will be read by our by parent's LSTM cell
            c.output.reads(state)
        return c

    def logits_and_state(self, word_idx):
        """Creates a block that goes from tokens to (logits, state) tuples."""
        unknown_idx = len(word_idx)
        lookup_word = lambda word: word_idx.get(word, unknown_idx)

        word2vec = (td.GetItem(0) >> td.InputTransform(lookup_word) >>
                    td.Scalar('int32') >> self.word_embedding)

        pair2vec = (self.embed_subtree(), self.embed_subtree())

        # Trees are binary, so the tree layer takes two states as its input_state.
        zero_state = td.Zeros((self.tree_lstm.state_size,) * 2)
        # Input is a word vector.
        zero_inp = td.Zeros(self.word_embedding.output_type.shape[0])

        word_case = td.AllOf(word2vec, zero_state)
        pair_case = td.AllOf(zero_inp, pair2vec)

        tree2vec = td.OneOf(len, [(1, word_case), (2, pair_case)])

        return tree2vec >> self.tree_lstm >> (self.output_layer, td.Identity())

    @staticmethod
    def tf_node_loss(logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)

    @staticmethod
    def tf_fine_grained_hits(logits, labels):
        predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
        return tf.cast(tf.equal(predictions, labels), tf.float64)

    @staticmethod
    def tf_binary_hits(logits, labels):
        softmax = tf.nn.softmax(logits)
        binary_predictions = (softmax[:, 3] + softmax[:, 4]) \
                             > (softmax[:, 0] + softmax[:, 1])
        binary_labels = labels > 2
        return tf.cast(tf.equal(binary_predictions, binary_labels), tf.float64)

    @staticmethod
    def embed_tree(logits_and_state, is_root):
        """Creates a block that embeds trees; output is tree LSTM state."""

        def tokenize(s):
            label, phrase = s[1:-1].split(None, 1)
            return label, sexpr.sexpr_tokenize(phrase)

        return td.InputTransform(tokenize) >> td.OneOf(
            key_fn=lambda pair: pair[0] == '2',  # label 2 means neutral
            case_blocks=(TreeLSTM.add_metrics(is_root, is_neutral=False),
                         TreeLSTM.add_metrics(is_root, is_neutral=True)),
            pre_block=(td.Scalar('int32'), logits_and_state))
