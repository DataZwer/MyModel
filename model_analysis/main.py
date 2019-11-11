from model_analysis.dmcnn_fc import dm_cnn_fc
from bean.Configuration import config
from bean.file_path import *
import tensorflow as tf

'''
twitter_path[twitter_train_path],
twitter_path[twitter_test_path],
twitter_path[twitter_emb_path]

restaurant_path[restaurant_train_path],
restaurant_path[restaurant_test_path],
restaurant_path[restaurant_emb_path],

laptop_path[laptop_train_path],
laptop_path[laptop_test_path],
laptop_path[laptop_emb_path],
'''


max_acc_temp = []
for i in range(5):
    tf.reset_default_graph()  # 重置当前计算图
    with tf.Session() as sess:
        model = dm_cnn_fc(
            config, sess,
            twitter_path[twitter_train_path],
            twitter_path[twitter_test_path],
            twitter_path[twitter_emb_path],
            layer2=True,
        )
        # model.sample_test()
        print("*******The %dth model*******" % (i+1))
        model.train()
        max_acc = max(model.all_test_acc)
        max_acc_temp.append(max_acc)
    tf.get_default_graph().finalize()  # 获取当前计算图并结束它

print(max_acc_temp)
print(max(max_acc_temp))
