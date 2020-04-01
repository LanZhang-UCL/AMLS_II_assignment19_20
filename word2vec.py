# This file is used to generate word2vec model for word embedding.
from gensim.models import Word2Vec
from pandas import read_csv
from keras.preprocessing.text import text_to_word_sequence

# Vocabulary appearing in training data and test data is used to train the model.
train_df = read_csv('./Datasets/my_training_data.csv')
test_df = read_csv('./Datasets/my_test_data.csv')
text = train_df['content']
text = text.append(test_df['content'], ignore_index=True)
for i in range(0, len(text)):
    text[i] = text_to_word_sequence(text[i])
my_word2vec = Word2Vec(text, size=200, min_count=2)
my_word2vec.wv.save('./word2vec.wv')