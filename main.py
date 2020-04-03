import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# ======================================================================================================================
# load test data and models
df = pd.read_csv('./Datasets/my_test_data.csv')
with open('./Tok.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
model_B = load_model('./B/B.h5')
model_D = load_model('./D/D.h5')


# ======================================================================================================================
# Test configure
max_seq_len = 200
batch_size = 128


# ======================================================================================================================
# Transform test data
x_test = tokenizer.texts_to_sequences(df['content'])
x_test = pad_sequences(x_test, maxlen=max_seq_len, padding='post', truncating='post')
y_test = df['label']


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
print('B_loss:{}, B_accuracy{}:, D_loss:{}, D_accuracy:{}'.format(loss_B, acc_B, loss_D, acc_D))

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