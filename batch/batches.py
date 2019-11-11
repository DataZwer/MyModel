import math
import tensorflow as tf


def dataset_iterator(word_up, positions_up, poses_up, sts_up, word_down,
                     positions_down, poses_down, sts_down, labels, data_len):
    batch_size = 64
    train_nums = data_len
    iterations = math.ceil(train_nums / batch_size)  # 总共可以划分出来的batch数量

    # 使用from_tensor_slices将数据放入队列，使用batch和repeat划分数据批次，且让数据序列无限延续
    dataset = tf.data.Dataset.from_tensor_slices((word_up, positions_up, poses_up, sts_up,
                                                  word_down, positions_down, poses_down, sts_down, labels))
    dataset = dataset.batch(batch_size).repeat()
    # dataset = dataset.batch(batch_size).repeat()

    # 使用生成器make_one_shot_iterator和get_next取数据
    iterator = dataset.make_one_shot_iterator()
    next_iterator = iterator.get_next()
    return iterations, next_iterator


#   train_data, test_data, all_word_index, max_len = \
#             load_all(restaurant_path[restaurant_train_path], restaurant_path[restaurant_test_path])
#
# iterations, next_iterator = dataset_iterator(
#             train_data[tr_sents_up],
#             train_data[tr_positions_up],
#             train_data[tr_pos_up],
#             train_data[tr_sts_up],
#             train_data[tr_sents_down],
#             train_data[tr_positions_down],
#             train_data[tr_pos_down],
#             train_data[tr_sts_down],
#             train_data[tr_labels],
#             len(train_data[tr_sents_up])
#         )
#
#
# with tf.Session() as sess:
#     for epoch in range(1):
#         for iteration in range(iterations):
#
#             # cu_image_batch, cu_label_batch = sess.run(next_iterator)
#             # print('The {0} epoch, the {1} iteration, current batch is {2}'.format(epoch + 1, iteration + 1, \
#             # cu_label_batch))
#
#             word_up_batch, position_up_batch, poses_up_batch, sts_up_batch, \
#             word_down_batch, position_down_batch, poses_down_batch, sts_down_batch, \
#             labels_batch = sess.run(next_iterator)
#             # print('The {0} epoch, the {1} iteration, current batch is {2}'.format(epoch + 1, iteration + 1, y_batch))
#             break
#         break
# print("hehe")
