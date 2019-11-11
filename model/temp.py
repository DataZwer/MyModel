file = open(r'D:\NLP程序相关\MyModel\model\acc_temp.txt', 'w', encoding='utf-8')
l = [0.1, 34, 123]
text = 'flajlfjaljfla'
file.writelines([str(e)+' ' for e in l] + ['\n'])
file.writelines(text + '\n')
file.writelines(str(max(l)))
file.close()


'''
def sample_test(self):
    self.opt_op()
    init = tf.global_variables_initializer()
    self.sess.run(init)
    saver = tf.train.Saver()
    model_file = tf.train.latest_checkpoint(r'D:\NLP程序相关\MyModel\model_saver')
    saver.restore(self.sess, model_file)

    def load_id(text, word_index, max_len):
        id = []
        for t_s in text:
            if t_s in word_index.keys():
                id.append(word_index[t_s])
            else:
                id.append(0)
        id = id + [0] * (max_len - len(id))
        return id
        # 创建和原来一样的网络

    up_text = "certainly not the best sushi in new york , however , it is always fresh , and the place".split(" ")
    down_text = "place is very clean, sterile .".split(" ")
    up_sents = "certainly not the best sushi in new york , however , it is always fresh , and the".split(" ")
    down_sents = "is very clean , sterile .".split(" ")
    aspects = "place".split(" ")

    word_up = load_id(up_text, self.all_word_index[word_index], self.max_len)
    word_down = load_id(down_text, self.all_word_index[word_index], self.max_len)
    positon_up, positon_down = load_postions([up_sents], [down_sents], [aspects])
    positon_up = load_id(
        positon_up[0],
        self.all_word_index[position_index],
        self.max_len
    )
    positon_down = load_id(
        positon_down[0],
        self.all_word_index[position_index],
        self.max_len
    )
    poses_up, poses_down = load_postions([up_sents], [down_sents], [aspects])
    poses_up = load_id(
        poses_up[0],
        self.all_word_index[poses_index],
        self.max_len
    )
    poses_down = load_id(
        poses_down[0],
        self.all_word_index[poses_index],
        self.max_len
    )
    sts_up, sts_down = load_sentiment_score([up_sents], [down_sents], [aspects])
    sts_up = load_id(
        sts_up[0],
        self.all_word_index[sts_index],
        self.max_len
    )
    sts_down = load_id(
        sts_down[0],
        self.all_word_index[sts_index],
        self.max_len
    )

    f_dict = {
        self.word_up: [word_up],
        self.positons_up: [word_down],
        self.poses_up: [positon_up],
        self.sts_up: [positon_down],
        self.word_down: [poses_up],
        self.positons_down: [poses_down],
        self.poses_down: [sts_up],
        self.sts_down: [sts_down],
        self.labels: [[0, 1, 0]],
        self.dropout_keep_prob: 1.0
    }

    loss, acc, att_res_up, att_res_down = self.sess.run(
        [self.loss_op, self.accuracy_op, self.alpha_up, self.alpha_down],
        feed_dict=f_dict
    )

    print([e[0] for e in att_res_up[0][:len(up_text) - len(aspects) + 1]])
    print([e[0] for e in att_res_down[0][len(aspects):len(down_text) + 1]])

'''


