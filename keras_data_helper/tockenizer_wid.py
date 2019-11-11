from keras.preprocessing.text import Tokenizer


def get_tokenizer_wid(tr_up, tr_down, te_up, te_down):
    text = tr_up + tr_down + te_up + te_down
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index
    return tokenizer, word_index


