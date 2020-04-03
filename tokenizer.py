import pickle
from pandas import read_csv
from keras.preprocessing.text import Tokenizer

df_train = read_csv('./Datasets/my_training_data.csv')
df_test = read_csv('./Datasets/my_test_data.csv')

dic = list(df_train['content'])
dic.extend(list(df_test['content']))

num_words = 20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(dic)

# Save Tokenizer
with open('./Tok.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)