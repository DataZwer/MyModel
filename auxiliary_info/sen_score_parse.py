
def data_parse(file_path):
    item_score = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.rstrip().split("\t")
            if line[0][0] == '#' or line[0][0] == '@':
                line[0] = line[0][1:]
            item_score[line[0]] = line[1]
    return item_score


item_score = data_parse(r"D:\NLP程序相关\MyModel\auxiliary_info\unigrams-pmilexicon.txt")

'''
import tensorflow as tf
import nltk as tk

if __name__ =="__main__":
    text = "good bad excellent"
    tokenizer = tk.WordPunctTokenizer()
    text = list(tokenizer.tokenize(text))  # 保留标点符号
    file_path = './unigrams-pmilexicon.txt'
    item_score = data_parse(file_path)
    text_map = {}
    for elem in text:
        if elem in item_score.keys():
            text_map[elem] = float(item_score[elem])
        else:
            text_map[elem] = 0.0
    print(text_map.values())

    with tf.Session() as sess:
        res = tf.nn.softmax(list(text_map.values()))
        print(sess.run(res))
'''
