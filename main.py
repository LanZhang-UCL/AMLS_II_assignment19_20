import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2VecKeyedVectors
from keras.preprocessing.text import text_to_word_sequence
from keras.models import load_model

# ======================================================================================================================
# load test data and models
df = pd.read_csv('./Datasets/my_test_data.csv')
wv = Word2VecKeyedVectors.load('./word2vec.wv')
model_B = load_model('./B/B.h5')
model_D = load_model('./D/D.h5')


# ======================================================================================================================
# Test configure
embedding_size = wv.vector_size
max_seq_len = 200
batch_size = 128


# ======================================================================================================================
# Obtain test data
x_test = np.zeros((len(df['content']), max_seq_len, embedding_size), dtype=np.float32)
y_test = df['label']
for i in range(0, len(df['content'])):
    temp = text_to_word_sequence(df['content'][i])
    for j in range(0, min(len(temp), max_seq_len)):
        if temp[j] in wv.vocab:
            x_test[i][j] = wv.get_vector(temp[j])


# ======================================================================================================================
# Models predict and evaluate
model_B.summary()
y_pred_B = model_B.predict(x_test, batch_size=batch_size).flatten()
loss_B, acc_B = model_B.evaluate(x_test, y_test, batch_size=batch_size)

model_D.summary()
y_pred_D = model_D.predict(x_test, batch_size=batch_size).flatten()
loss_D, acc_D = model_D.evaluate(x_test, y_test, batch_size=batch_size)


# ======================================================================================================================
# print results
print('B_loss:{}, B_accuracy{}, D_loss:{}, D_accuracy:{}'.format(loss_B, acc_B, loss_D, acc_D))

# ======================================================================================================================
# write prediction to .csv file
B_df = {'id': df['id'], 'topic': df['topic'], 'label': y_test,
        'predict_label': (y_pred_B > 0.5).astype(np.int32), 'score': y_pred_B}
B_df = pd.DataFrame(B_df)
B_df.to_csv('./B.csv', index=False)
D_df = {'id': df['id'], 'topic': df['topic'], 'label': y_test,
        'predict_label': (y_pred_D > 0.5).astype(np.int32), 'score': y_pred_D}
D_df = pd.DataFrame(D_df)
D_df.to_csv('./D.csv', index=False)