import sys
import numpy
import tensorflow as tf

TRAIN_TIME_MINUTES = 11


class DatasetReader(object):
    @staticmethod
    def ReadFile(filename, term_index, tag_index):
        """Reads file into dataset, while populating term_index and tag_index.

        Args:
          filename: Path of text file containing sentences and tags. Each line is a
            sentence and each term is followed by "/tag". Note: some terms might
            have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
            separates the tag.
          term_index: dictionary to be populated with every unique term (i.e. before
            the last "/") to point to an integer. All integers must be utilized from
            0 to number of unique terms - 1, without any gaps nor repetitions.
          tag_index: same as term_index, but for tags.

        Return:
          The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
          each parsedLine is a list: [(term1, tag1), (term2, tag2), ...]
        """
        with open(filename, "r", encoding="utf8") as filename:
            lines = []
            for line in filename:
                lines.append(line)

            parsed_file = []
            for line in lines:
                words_tags = line.split()
                parsed_line = []
                for pair in words_tags:
                    temp = pair.split("/")
                    word = "/".join(temp[:-1])
                    tag = temp[-1]
                    if word not in term_index:
                        term_index[word] = len(term_index)
                    if tag not in tag_index:
                        tag_index[tag] = len(tag_index)
                    parsed_line.append((term_index[word], tag_index[tag]))
                parsed_file.append(parsed_line)
            return parsed_file


    @staticmethod
    def BuildMatrices(dataset):
        """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.

        Args:
          dataset: Returned by method ReadFile. It is a list (length N) of lists:
            [sentence1, sentence2, ...], where every sentence is a list:
            [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.

        Returns:
          Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
            terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
              indices in dataset[i].
            tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
              indices in dataset[i].
            lengths: shape (N) int64 numpy array. Entry i contains the length of
              sentence in dataset[i].

          T is the maximum length. For example, calling as:
            BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
          i.e. with two sentences, first with length 2 and second with length 4,
          should return the tuple:
          (
            [[1, 4, 0, 0],    # Note: 0 padding.
             [13, 3, 7, 3]],

            [[2, 10, 0, 0],   # Note: 0 padding.
             [20, 6, 8, 20]],

            [2, 4]
          )
        """

        n = len(dataset)
        t = len(max(dataset, key=lambda x: len(x)))
        terms_matrix = numpy.zeros(n * t, dtype='int64')
        terms_matrix = terms_matrix.reshape(n, t)
        tags_matrix = numpy.zeros(n * t, dtype='int64')
        tags_matrix = tags_matrix.reshape(n, t)
        lengths = numpy.zeros(n, dtype='int64')

        for i in range(len(dataset)):
            for j in range(len(dataset[i])):
                terms_matrix[i, j] = dataset[i][j][0]
                tags_matrix[i, j] = dataset[i][j][1]
                lengths[i] = len(dataset[i])
        return terms_matrix, tags_matrix, lengths

    @staticmethod
    def ReadData(train_filename, test_filename=None):
        """Returns numpy arrays and indices for train (and optionally test) data.

        NOTE: Please do not change this method. The grader will use an identitical
        copy of this method (if you change this, your offline testing will no longer
        match the grader).

        Args:
          train_filename: .txt path containing training data, one line per sentence.
            The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
          test_filename: Optional .txt path containing test data.

        Returns:
          A tuple of 3-elements or 4-elements, the later iff test_filename is given.
          The first 2 elements are term_index and tag_index, which are dictionaries,
          respectively, from term to integer ID and from tag to integer ID. The int
          IDs are used in the numpy matrices.
          The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
            - train_terms: numpy int matrix.
            - train_tags: numpy int matrix.
            - train_lengths: numpy int vector.
            These 3 are identical to what is returned by BuildMatrices().
          The 4th element is a tuple of 3 elements as above, but the data is
          extracted from test_filename.
        """
        term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
        tag_index = {}

        train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
        train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)

        if test_filename:
            test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
            test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

            if test_tags.shape[1] < train_tags.shape[1]:
                diff = train_tags.shape[1] - test_tags.shape[1]
                zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
                test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
                test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
            elif test_tags.shape[1] > train_tags.shape[1]:
                diff = test_tags.shape[1] - train_tags.shape[1]
                zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
                train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
                train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

            return (term_index, tag_index,
                    (train_terms, train_tags, train_lengths),
                    (test_terms, test_tags, test_lengths))
        else:
            return term_index, tag_index, (train_terms, train_tags, train_lengths)


class SequenceModel(object):

    def __init__(self, max_length=310, num_terms=1000, num_tags=40):
        """
        Constructor

        The arguments are arbitrary.
        Args:
          max_length: maximum possible sentence length.
          num_terms: the vocabulary size (number of terms).
          num_tags: the size of the output space (number of tags).

        """
        self.sess = None
        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags
        self.x = tf.placeholder(tf.int64, [None, self.max_length], name='x')
        self.y = tf.placeholder(tf.int64, [None, self.max_length], name='y')
        self.logits = None
        self.lengths = tf.placeholder(tf.int32, [None], name='lengths')
        self.learning_rate = tf.placeholder_with_default(
            numpy.array(0.005, dtype='float32'), shape=[], name='learn_rate')
        self.train_op = None


    def lengths_vector_to_binary_matrix(self, length_vector):
        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.

        Specifically, the return matrix B will have the following:
          B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """
        # return tf.ones([tf.shape(length_vector), self.max_length], dtype=tf.float32)
        return tf.sequence_mask(length_vector, maxlen=self.max_length, dtype=tf.float32)


    def save_model(self, filename):
        """Saves model to a file."""
        pass

    # TODO(student): You must implement this.
    def load_model(self, filename):
        """Loads model from a file."""
        pass

    # TODO(student): You must implement this.
    def build_inference(self):
        """Build the expression from (self.x, self.lengths) to (self.logits).

        Please do not change or override self.x nor self.lengths in this function.

        """
        l2_reg_val = 1e-6
        num_cells = 128
        embeddings = tf.get_variable("embeddings", [self.num_terms, 40])
        x_embeddings = tf.nn.embedding_lookup(embeddings, self.x)

        cells_fw = tf.nn.rnn_cell.BasicLSTMCell(num_cells, forget_bias=1.0)
        cells_fw = tf.contrib.rnn.DropoutWrapper(cells_fw, output_keep_prob=0.8)
        cells_bw = tf.nn.rnn_cell.BasicLSTMCell(num_cells, forget_bias=1.0)
        cells_bw = tf.contrib.rnn.DropoutWrapper(cells_bw, output_keep_prob=0.8)

        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([cells_fw],
                                                                                                   [cells_bw],
                                                                                                   x_embeddings,
                                                                                                   dtype=tf.float32)
        l2_reg = tf.contrib.layers.l2_regularizer(l2_reg_val)
        self.logits = tf.contrib.layers.fully_connected(outputs, self.num_tags, activation_fn=None,
                                                        weights_regularizer=l2_reg)


    def run_inference(self, terms, lengths):
        """Evaluates self.logits given self.x and self.lengths.

        Hint: This function is straight forward and you might find this code useful:
        # logits = session.run(self.logits, {self.x: terms, self.lengths: lengths})
        # return numpy.argmax(logits, axis=2)

        Args:
          terms: numpy int matrix, like terms_matrix made by BuildMatrices.
          lengths: numpy int vector, like lengths made by BuildMatrices.

        Returns:
          numpy int matrix of the predicted tags, with shape identical to the int
          matrix tags i.e. each term must have its associated tag. The caller will
          *not* process the output tags beyond the sentence length i.e. you can have
          arbitrary values beyond length.
        """
        # return numpy.zeros_like(tags)
        logits = self.sess.run(self.logits, {self.x: terms, self.lengths: lengths})
        return numpy.argmax(logits, axis=2)


    def build_training(self):
        """Prepares the class for training.

        It is up to you how you implement this function, as long as train_epoch
        works.

        Hint:
          - Lookup tf.contrib.seq2seq.sequence_loss
          - tf.losses.get_total_loss() should return a valid tensor (without raising
            an exception). Equivalently, tf.losses.get_losses() should return a
            non-empty list.
        """
        seq2seq_sequence_loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.y,
                                                                 weights=self.lengths_vector_to_binary_matrix(
                                                                     self.lengths))
        tf.losses.add_loss(seq2seq_sequence_loss)   # add loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)    # optimizer
        self.train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), optimizer)  #train op
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train_epoch(self, terms, tags, lengths, batch_size=32, learn_rate=0.005):
        """Performs updates on the model given training training data.

        This will be called with numpy arrays similar to the ones created in
        Args:
          terms: int64 numpy array of size (# sentences, max sentence length)
          tags: int64 numpy array of size (# sentences, max sentence length)
          lengths:
          batch_size: int indicating batch size. Grader script will not pass this,
            but it is only here so that you can experiment with a "good batch size"
            from your main block.
          learn_rate: float for learning rate. Grader script will not pass this,
            but it is only here so that you can experiment with a "good learn rate"
            from your main block.
        """
        print("Training Model...")
        self.step(terms, tags, lengths, batch_size, learn_rate)
        return True

    def batch_step(self, batch_x, batch_y, batch_lengths, learn_rate):
        self.sess.run(self.train_op, {
            self.lengths: batch_lengths,
            self.x: batch_x,
            self.y: batch_y,
            self.learning_rate: learn_rate
        })

    def step(self, terms, tags, lengths, batch_size, learn_rate):
        indices = numpy.random.permutation(terms.shape[0])

        for si in range(0, terms.shape[0], batch_size):
            se = min(si + batch_size, terms.shape[0])
            slice_x = terms[indices[si:se]] + 0
            slice_lengths = lengths[indices[si:se]] + 0
            self.batch_step(slice_x, tags[indices[si:se]], slice_lengths, learn_rate)


def main():
    # Read dataset.
    reader = DatasetReader
    train_filename = sys.argv[1]
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data

    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    for j in range(10):
        model.train_epoch(train_terms, train_tags, train_lengths)
        print('Finished epoch %i. Evaluating ...' % (j + 1))
        # model.evaluate(test_terms, test_tags, test_lengths)


if __name__ == '__main__':
    main()
