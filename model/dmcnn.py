from keras_data_helper.load_embedding import load_glove_embedding, load_random_embedding
from keras_data_helper.data_utils import load_all
from bean.file_path import *
from bean.Configuration import *
from batch.batches import dataset_iterator
import tensorflow as tf


class dm_cnn(object):
    def __init__(self, config, sess, train_path, test_path, embedding_path):
        # 模型常量定义
        self.n_epochs = config['n_epochs']
        self.filter_sizes = config['filter_sizes']
        self.embedding_size = config['embedding_size']
        self.num_filters = config['num_filters']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.num_classes = config['num_classes']
        self.batch_size = config['batch_size']
        self.sess = sess
        self.l2_loss = tf.constant(0.0)
        self.all_test_acc = []
        self.train_path = train_path
        self.test_path = test_path
        self.embedding_path = embedding_path

        # input
        self.train_data = None
        self.test_data = None
        self.all_word_index = None
        self.max_len = None
        self.word_up = None
        self.word_down = None
        self.positons_up = None
        self.positons_down = None
        self.sts_up = None
        self.sts_down = None
        self.poses_up = None
        self.poses_down = None
        self.labels = None
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.final_res = None

        # opt
        self.score = None
        self.prediction = None
        self.losses = None
        self.correct_predictions = None
        self.loss_op = None
        self.accuracy_op = None
        self.optim = None
        self.train_op = None

        # embedding
        self.positions_up_embed = None
        self.positions_down_embed = None
        self.poses_up_embed = None
        self.poses_down_embed = None
        self.sts_up_embed = None
        self.sts_down_embed = None
        self.word_up_embed = None
        self.word_down_embed = None
        self.up_compose = None
        self.down_compose = None

        # embedding

    def input_layer(self):
        self.train_data, self.test_data, self.all_word_index, self.max_len = \
            load_all(self.train_path, self.test_path)
        self.word_up = tf.placeholder(tf.int32, [None, self.max_len], name='word_up')
        self.word_down = tf.placeholder(tf.int32, [None, self.max_len], name='word_down')
        self.positons_up = tf.placeholder(tf.int32, [None, self.max_len], name='positons_up')
        self.positons_down = tf.placeholder(tf.int32, [None, self.max_len], name='positons_down')
        self.sts_up = tf.placeholder(tf.int32, [None, self.max_len], name='sts_up')
        self.sts_down = tf.placeholder(tf.int32, [None, self.max_len], name='sts_down')
        self.poses_up = tf.placeholder(tf.int32, [None, self.max_len], name='poses_up')
        self.poses_down = tf.placeholder(tf.int32, [None, self.max_len], name='poses_down')
        self.labels = tf.placeholder(tf.float32, [None, 3], name='labels')

    def embedding_layer(self):
        self.input_layer()
        # 随机初始化
        self.positions_up_embed = load_random_embedding(len(self.all_word_index[position_index]), self.embedding_size, self.positons_up)
        self.positions_down_embed = load_random_embedding(len(self.all_word_index[position_index]), self.embedding_size, self.positons_down)
        self.poses_up_embed = load_random_embedding(len(self.all_word_index[poses_index]), self.embedding_size, self.poses_up)
        self.poses_down_embed = load_random_embedding(len(self.all_word_index[poses_index]), self.embedding_size, self.poses_down)
        self.sts_up_embed = load_random_embedding(len(self.all_word_index[poses_index]), self.embedding_size, self.sts_up)
        self.sts_down_embed = load_random_embedding(len(self.all_word_index[poses_index]), self.embedding_size, self.sts_down)

        # glove
        embedding_maxtrix = load_glove_embedding(
            word_index=self.all_word_index[word_index],
            file='',
            trimmed_filename=self.embedding_path,
            load=True,
            dim=self.embedding_size
        )
        glove_w2v = tf.Variable(embedding_maxtrix, dtype=tf.float32, name='glove_w2v')
        self.word_up_embed = tf.nn.embedding_lookup(glove_w2v, self.word_up)
        self.word_down_embed = tf.nn.embedding_lookup(glove_w2v, self.word_down)

        # dropout [b_s, max_len, 300, 1]
        self.word_up_embed = tf.nn.dropout(tf.expand_dims(self.word_up_embed, -1), self.dropout_keep_prob)
        self.word_down_embed = tf.nn.dropout(tf.expand_dims(self.word_down_embed, -1), self.dropout_keep_prob)
        self.positions_up_embed = tf.nn.dropout(tf.expand_dims(self.positions_up_embed, -1), self.dropout_keep_prob)
        self.positions_down_embed = tf.nn.dropout(tf.expand_dims(self.positions_down_embed, -1), self.dropout_keep_prob)
        self.poses_up_embed = tf.nn.dropout(tf.expand_dims(self.poses_up_embed, -1), self.dropout_keep_prob)
        self.poses_down_embed = tf.nn.dropout(tf.expand_dims(self.poses_down_embed, -1), self.dropout_keep_prob)
        self.sts_up_embed = tf.nn.dropout(tf.expand_dims(self.sts_up_embed, -1), self.dropout_keep_prob)
        self.sts_down_embed = tf.nn.dropout(tf.expand_dims(self.sts_down_embed, -1), self.dropout_keep_prob)

        # 合成 [b_s, max_len, 300, 1]
        self.up_compose = tf.add(
            tf.add(self.word_up_embed, self.positions_up_embed),
            tf.add(self.sts_up_embed, self.poses_up_embed)
        )
        self.down_compose = tf.add(
            tf.add(self.word_down_embed, self.positions_down_embed),
            tf.add(self.sts_down_embed, self.poses_down_embed)
        )

    def conv2d_op(self, embedding, max_len):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='conv2d_filer')
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='conv2d_b')
            conv = tf.nn.conv2d(
                embedding,
                W, strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv'
            )
            # activation
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            # max pooling
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_len - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool'
            )
            pooled_outputs.append(pooled)
        # 全连接
        h_pool = tf.concat(pooled_outputs, axis=3)  # [b_s, 1, 1, 300]
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])  # [b_s, 300]
        return h_pool_flat

    def dmcnn_pooling(self):
        self.embedding_layer()
        up_conv_pool = self.conv2d_op(self.up_compose, int(self.up_compose.shape[1]))
        down_conv_pool = self.conv2d_op(self.down_compose, int(self.down_compose.shape[1]))
        # [b_s, 600]
        self.final_res = tf.nn.dropout(tf.concat([up_conv_pool, down_conv_pool], axis=1), self.dropout_keep_prob)

    def softmax_output(self):
        self.dmcnn_pooling()
        W = tf.get_variable(
            name='W',
            shape=[self.num_filters * len(self.filter_sizes) * 2, self.num_classes],  # 有两边
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')

        self.l2_loss += tf.nn.l2_loss(W)
        self.l2_loss += tf.nn.l2_loss(b)

        self.score = tf.nn.xw_plus_b(self.final_res, W, b, name='scores')
        self.prediction = tf.argmax(self.score, 1, name='prediction')

    def opt_op(self):
        self.softmax_output()
        self.losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.score,
            labels=self.labels
        )

        self.loss_op = tf.reduce_mean(self.losses) + self.l2_reg_lambda * self.l2_loss  # l2 正则化
        self.correct_predictions = tf.equal(self.prediction, tf.argmax(self.labels, 1))
        self.accuracy_op = tf.reduce_mean(
            tf.cast(self.correct_predictions, 'float'),
            name='accuracy'
        )
        self.optim = tf.train.AdamOptimizer(learning_rate=0.001)  # Adam优化器
        self.train_op = self.optim.minimize(self.loss_op)  # 使用优化器最小化损失函数

    def train(self):
        self.opt_op()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # 划分数据batch
        iterations, next_iterator = dataset_iterator(
            self.train_data[tr_sents_up],
            self.train_data[tr_positions_up],
            self.train_data[tr_pos_up],
            self.train_data[tr_sts_up],
            self.train_data[tr_sents_down],
            self.train_data[tr_positions_down],
            self.train_data[tr_pos_down],
            self.train_data[tr_sts_down],
            self.train_data[tr_labels],
            len(self.train_data[tr_sents_up])
        )
        for epoch in range(self.n_epochs):
            count = 0
            train_loss = 0
            train_acc = 0
            print("-----------Now we begin the %dth epoch-----------" % (epoch))
            for iter in range(iterations):
                word_up_batch, position_up_batch, poses_up_batch, sts_up_batch, \
                word_down_batch, position_down_batch, poses_down_batch, sts_down_batch, \
                labels_batch = self.sess.run(next_iterator)

                if len(word_up_batch) < self.batch_size:
                    continue

                f_dict = {
                    self.word_up: word_up_batch,
                    self.positons_up: position_up_batch,
                    self.poses_up: poses_up_batch,
                    self.sts_up: sts_up_batch,
                    self.word_down: word_down_batch,
                    self.positons_down: position_down_batch,
                    self.poses_down: poses_down_batch,
                    self.sts_down: sts_down_batch,
                    self.labels: labels_batch,
                    self.dropout_keep_prob: 0.5
                }

                _, loss, acc = self.sess.run([self.train_op, self.loss_op, self.accuracy_op], feed_dict=f_dict)
                train_loss = train_loss+loss
                train_acc = train_acc+acc
                count = count + 1
            train_loss = train_loss / count
            train_acc = train_acc / count
            print("-----------After the %dth epoch, the train loss is: %f, the train acc is: %f-----------" % (epoch, train_loss, train_acc))

            # test
            iterations_test, next_iterator_test = dataset_iterator(
                self.test_data[te_sents_up],
                self.test_data[te_positions_up],
                self.test_data[te_pos_up],
                self.test_data[te_sts_up],
                self.test_data[te_sents_down],
                self.test_data[te_positions_down],
                self.test_data[te_pos_down],
                self.test_data[te_sts_down],
                self.test_data[te_labels],
                len(self.test_data[te_sents_up])
            )
            self.test(iterations_test, next_iterator_test, epoch, self.loss_op, self.accuracy_op)

    def test(self, iterations_test, next_iterator_test, epoch, loss_op, accuracy_op):
        test_loss = 0
        test_acc = 0
        count = 0

        for iter in range(iterations_test):
            word_up_batch, position_up_batch, poses_up_batch, sts_up_batch, \
            word_down_batch, position_down_batch, poses_down_batch, sts_down_batch, \
            labels_batch = self.sess.run(next_iterator_test)

            if len(word_up_batch) < self.batch_size:
                continue

            f_dict = {
                self.word_up: word_up_batch,
                self.positons_up: position_up_batch,
                self.poses_up: poses_up_batch,
                self.sts_up: sts_up_batch,
                self.word_down: word_down_batch,
                self.positons_down: position_down_batch,
                self.poses_down: poses_down_batch,
                self.sts_down: sts_down_batch,
                self.labels: labels_batch,
                self.dropout_keep_prob: 1.0
            }

            count = count + 1
            loss, acc = self.sess.run([loss_op, accuracy_op], feed_dict=f_dict)
            test_loss = test_loss + loss
            test_acc = test_acc+acc

        test_loss = test_loss / count
        test_acc = test_acc / count
        self.all_test_acc.append(test_acc)
        print("-----------After the %dth epoch, the test loss is: %f, the test acc is: %f-----------" % (epoch, test_loss, test_acc))