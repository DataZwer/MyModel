from keras_data_helper.load_embedding import load_glove_embedding, load_random_embedding
from keras_data_helper.data_utils import load_all
from bean.Configuration import *
from batch.batches import dataset_iterator
import tensorflow as tf


class dm_cnn_fc(object):
    def __init__(self, config, sess, train_path, test_path, embedding_path, layer2=False):
        # 模型常量定义
        self.n_epochs = config['n_epochs']
        self.filter_sizes = config['filter_sizes']
        self.embedding_size = config['embedding_size']
        self.num_filters = config['num_filters']
        self.l2_reg_lambda = config['self.l2_reg_lambda']
        self.num_classes = config['num_classes']
        self.batch_size = config['batch_size']
        self.ex_emb_size = config['ex_emb_size']
        self.train_path = train_path
        self.test_path = test_path
        self.embedding_path = embedding_path
        self.sess = sess
        self.l2_loss = tf.constant(0.0)
        self.all_test_acc = []
        self.layer2 = layer2

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
        self.up_connect = None
        self.down_connect = None

        # att
        self.alpha = None
        self.alpha_up = None
        self.alpha_down = None

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
        # 随机初始化 [b_s, max_len, 50]
        self.positions_up_embed = load_random_embedding(len(self.all_word_index[position_index]), self.ex_emb_size, self.positons_up)
        self.positions_down_embed = load_random_embedding(len(self.all_word_index[position_index]), self.ex_emb_size, self.positons_down)
        self.poses_up_embed = load_random_embedding(len(self.all_word_index[poses_index]), self.ex_emb_size, self.poses_up)
        self.poses_down_embed = load_random_embedding(len(self.all_word_index[poses_index]), self.ex_emb_size, self.poses_down)
        self.sts_up_embed = load_random_embedding(len(self.all_word_index[poses_index]), self.ex_emb_size, self.sts_up)
        self.sts_down_embed = load_random_embedding(len(self.all_word_index[poses_index]), self.ex_emb_size, self.sts_down)

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

        self.word_up_embed = tf.nn.dropout(self.word_up_embed, self.dropout_keep_prob)
        self.word_down_embed = tf.nn.dropout(self.word_down_embed, self.dropout_keep_prob)
        self.positions_up_embed = tf.nn.dropout(self.positions_up_embed, self.dropout_keep_prob)
        self.positions_down_embed = tf.nn.dropout(self.positions_down_embed, self.dropout_keep_prob)
        self.poses_up_embed = tf.nn.dropout(self.poses_up_embed, self.dropout_keep_prob)
        self.poses_down_embed = tf.nn.dropout(self.poses_down_embed, self.dropout_keep_prob)
        self.sts_up_embed = tf.nn.dropout(self.sts_up_embed, self.dropout_keep_prob)
        self.sts_down_embed = tf.nn.dropout(self.sts_down_embed, self.dropout_keep_prob)

        # 连接 [b_s, max_len, 300+3*ex_emb_size]
        self.up_connect = tf.concat(
            [self.word_up_embed, self.positions_up_embed, self.poses_up_embed, self.sts_up_embed],
            axis=2
        )
        self.down_connect = tf.concat(
            [self.word_down_embed, self.positions_down_embed, self.poses_down_embed, self.sts_up_embed],
            axis=2
        )

    def att_embed(self, batch_embedding):
        # 计算注意力: [b_s, max_len, 300+3*ex_emb_size] ,
        att_filter = tf.Variable(
            tf.truncated_normal(
                [5, self.embedding_size+3*self.ex_emb_size, 1],
                stddev=0.1),
            name='att_filter'
        )
        att_bias = tf.Variable(tf.truncated_normal([1], stddev=0.1), name='att_bias')
        att_conv = tf.nn.conv1d(batch_embedding, att_filter, 1, "SAME") + att_bias
        self.alpha = tf.nn.sigmoid(att_conv)  # [b_s, max_len, 1]

        # 词向量加权
        batch_embedding_weighted = tf.multiply(self.alpha, batch_embedding)
        return batch_embedding_weighted  # [b_s, max_len, 300+3*ex_emb_size]

    def att_res_compute(self, batch_embedding):
        batch_embedding_weighted = self.att_embed(batch_embedding)
        y = tf.reduce_sum(batch_embedding_weighted, 1, keep_dims=True)  # [b_s, 1, 300]

        att_res_filter = tf.Variable(
            tf.truncated_normal(
                [1, self.embedding_size+3*self.ex_emb_size, 400],
                stddev=0.1
            ),
            name='att_res_filter'
        )
        att_res_bias = tf.Variable(tf.truncated_normal([400], stddev=0.1), name='att_bias')
        att_res_conv = tf.nn.conv1d(y, att_res_filter, 1, "SAME") + att_res_bias
        att_res = tf.nn.tanh(att_res_conv)

        return att_res  # [b_s, 1, 400]

    def conv1d_op(self, embedding):
        new_size = self.embedding_size+3*self.ex_emb_size
        filter_shape = [3, new_size, new_size]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='conv1d_filer')
        b = tf.Variable(tf.constant(0.1, shape=[new_size]), name='conv1d_b')
        # conv layer
        # [b_s, max_len, new_size]
        sents_feature = tf.nn.relu(tf.nn.bias_add(tf.nn.conv1d(embedding, W, 1, "SAME"), b))
        # sents_feature =
        return sents_feature

    def conv2d_op(self, embedding, max_len):
        new_size = self.embedding_size + 3 * self.ex_emb_size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, new_size, 1, self.num_filters]
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

        if self.layer2 == True:
            # 第2层
            self.up_connect = self.conv1d_op(self.up_connect)
            self.down_connect = self.conv1d_op(self.down_connect)

        # 加权之后的embedding
        up_res = tf.squeeze(self.att_res_compute(self.up_connect), axis=[1])  # [b_s, 400]
        self.alpha_up = self.alpha
        down_res = tf.squeeze(self.att_res_compute(self.down_connect), axis=[1])  # [b_s, 1, 400]
        self.alpha_down = self.alpha

        # conv2d
        self.up_connect = tf.expand_dims(self.up_connect, -1)
        self.down_connect = tf.expand_dims(self.down_connect, -1)
        up_conv_pool = self.conv2d_op(self.up_connect, int(self.up_connect.shape[1]))
        down_conv_pool = self.conv2d_op(self.down_connect, int(self.down_connect.shape[1]))

        # 连接最终的结果
        up_concat = tf.concat([up_conv_pool, up_res], axis=1)  # [b_s, 700]
        down_concat = tf.concat([down_conv_pool, down_res], axis=1)  # [b_s, 700]
        # [b_s, 1400]
        self.final_res = tf.nn.dropout(tf.concat([up_concat, down_concat], axis=1), self.dropout_keep_prob)

    def softmax_output(self):
        self.dmcnn_pooling()
        W = tf.get_variable(
            name='W',
            shape=[(self.num_filters * len(self.filter_sizes) + 400) * 2, self.num_classes],  # 有两边
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
        saver = tf.train.Saver()

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
        # max_test_acc = 0
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

                _, loss, acc, att_up, att_down = self.sess.run([self.train_op, self.loss_op, self.accuracy_op, self.alpha_up, self.alpha_down], feed_dict=f_dict)
                train_loss = train_loss+loss
                train_acc = train_acc+acc
                count = count + 1

                if iter == 2:
                    print(str(att_up.shape) + str(att_down.shape))
                    print("up att: ", [float("%.3f" % e[0]) for e in att_up[3][:]])
                    print("%d up id: ", [e for e in word_up_batch[3][:]])
                    print("%d down att: ", [float("%.3f" % e[0]) for e in att_down[3][:]])
                    print("%d down id: ", [e for e in word_down_batch[3][:]])

            train_loss = train_loss / count
            train_acc = train_acc / count
            print("---%dth epoch train loss: %f, train acc: %f---" % (epoch, train_loss, train_acc))

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

            #     saver.save(self.sess, r'D:\NLP程序相关\MyModel\model_saver\dmcnn', global_step=epoch+1)
            #     max_test_acc = test_acc

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
            loss, acc, att_res_up, att_res_down = self.sess.run(
                [loss_op, accuracy_op, self.alpha_up, self.alpha_down],
                feed_dict=f_dict
            )
            test_loss = test_loss + loss
            test_acc = test_acc + acc

            # print([float("%.3f" % e[0]) for e in att_res_up[0][:]])
            # print([e for e in word_up_batch[0][:]])
            # print([float("%.3f" % e[0]) for e in att_res_down[0][:]])
            # print([e for e in word_down_batch[0][:]])

            # if iter == 0:
            #     print(str(att_res_up.shape) + str(att_res_down.shape))
            #     print("%d up att: ", [float("%.3f" % e[0]) for e in att_res_up[16][:]])
            #     print("%d up id: ", [e for e in word_up_batch[16][:]])
            #     print("%d down att: ", [float("%.3f" % e[0]) for e in att_res_down[16][:]])
            #     print("%d down id: ", [e for e in word_down_batch[16][:]])

                # for i in range(10):
                #     print("%d up att: " % i, [float("%.3f" % e[0]) for e in att_res_up[i][:]])
                #     print("%d up id: " % i, [e for e in word_up_batch[i][:]])
                #     print("%d down att: " % i, [float("%.3f" % e[0]) for e in att_res_down[i][:]])
                #     print("%d down id: " % i, [e for e in word_down_batch[i][:]])

        test_loss = test_loss / count
        test_acc = test_acc / count
        self.all_test_acc.append(test_acc)
        print("---%dth epoch test loss: %f, test acc: %f---" % (epoch, test_loss, test_acc))
