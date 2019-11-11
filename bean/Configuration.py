tr_labels = 'tr_labels'
te_labels = 'te_labels'

tr_sents_up = 'tr_sents_up'
tr_sents_down = 'tr_sents_down'
te_sents_up = 'te_sents_up'
te_sents_down = 'te_sents_down'

tr_positions_up = 'tr_positions_up'
tr_positions_down = 'tr_positions_down'
te_positions_up = 'te_positions_up'
te_positions_down = 'te_positions_down'

tr_pos_up = 'tr_pos_up'
tr_pos_down = 'tr_pos_down'
te_pos_up = 'te_pos_up'
te_pos_down = 'te_pos_down'

tr_sts_up = 'tr_sts_up'
tr_sts_down = 'tr_sts_down'
te_sts_up = 'te_sts_up'
te_sts_down = 'te_sts_down'

tr_aspects = 'tr_aspects'
te_aspects = 'te_aspects'

word_index = "word_index"
position_index = "position_index"
poses_index = "poses_index"
sts_index = "sts_index"

# train_up_max_len = 'train_up_max_len'
# train_down_max_len = 'train_down_max_len'
# test_up_max_len = 'test_up_max_len'
# test_down_max_len = 'test_down_max_len'

config = {
    'n_epochs': 30,
    'filter_sizes': [3, 4, 5],
    'num_filters': 100,
    'dropout_rate': 0.5,
    'learning_rate': 0.01,
    'std_dev': 0.05,
    'num_classes': 3,
    'l2_reg_lambda': 0.5,
    'batch_size': 64,
    'embedding_size': 300,
    'ex_emb_size': 50,
    'lstm_units': 300
}




