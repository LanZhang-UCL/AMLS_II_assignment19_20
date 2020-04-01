import os
import random
import numpy as np
from pandas import read_csv
from gensim.models.word2vec import Word2VecKeyedVectors
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Model
from keras.layers import Masking, LSTM, Dense, Input, Dropout
from keras.optimizers import RMSprop
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ======================================================================================================================
# load data and word embedding model
df = read_csv('../Datasets/my_training_data.csv')
wv = Word2VecKeyedVectors.load('../word2vec.wv')


# ======================================================================================================================
# Training configure
embedding_size = wv.vector_size
max_seq_len = 200
batch_size = 128
epochs = 50
learning_rate = 0.01


# ======================================================================================================================
# Obtain training data
X = np.zeros((len(df['content']), max_seq_len, embedding_size), dtype=np.float32)
Y = df['label']
for i in range(0, len(df['content'])):
    temp = text_to_word_sequence(df['content'][i])
    for j in range(0, min(len(temp), max_seq_len)):
        if temp[j] in wv.vocab:
            X[i][j] = wv.get_vector(temp[j])


# ======================================================================================================================
# Train-validation split
def data_split(x, y, train_proportion=0.8, random_state=0):
    # This function is used to split data into training set and test set.

    x_samples, timesteps, features = x.shape
    y_samples = y.shape[0]
    if x_samples != y_samples:
        print("Samples do not match, please correct your data!")
        os._exit(1)
    else:
        n_samples = x_samples
    n_train = int(n_samples * train_proportion)
    n_test = n_samples - n_train
    i = list(range(0, n_samples))
    if random_state != 0:
        random.seed(random_state)
    i_train = random.sample(i, k=n_train)
    for a in i_train:
        i.remove(a)
    i_test = i
    x_train = np.empty((n_train, timesteps, features), dtype=np.float32)
    y_train = np.empty(n_train, dtype=int)
    for a in range(0, n_train):
        x_train[a] = x[i_train[a]]
        y_train[a] = y[i_train[a]]
    x_test = np.empty((n_test, timesteps, features), dtype=np.float32)
    y_test = np.empty(n_test, dtype=int)
    for a in range(0, n_test):
        x_test[a] = x[i_test[a]]
        y_test[a] = y[i_test[a]]
    print('successfully split data')
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = data_split(X, Y, random_state=3)


# ======================================================================================================================
# Build Model
input = Input(shape=(max_seq_len, embedding_size))
B1 = Masking(mask_value=0.0)(input)
B2 = LSTM(units=128)(B1)
B3 = Dropout(0.5)(B2)
B4 = Dense(32)(B3)
output = Dense(1, activation='sigmoid')(B4)

model = Model(input=input, output=output)
model.summary()


# ======================================================================================================================
# Compile and train model
model.compile(optimizer=RMSprop(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(x_test, y_test),
                 callbacks=[
                     EarlyStopping(
                         monitor='val_loss', patience=6,
                         restore_best_weights=True, min_delta=0.0001),
                     ReduceLROnPlateau(
                         monitor='val_loss', factor=0.3, patience=3,
                         min_delta=0.0001, min_lr=0.0001)])
model.save('B.h5')


# ======================================================================================================================
# Visualization
# Plot learning rate & train accuracy
plt.plot(hist.history['lr'], hist.history['accuracy'])
plt.plot(hist.history['lr'], hist.history['val_accuracy'])
plt.title('Hyperparameter')
plt.xlim(0, learning_rate*1.1)
plt.ylim(0.7, 1)
plt.ylabel('Train Accuracy')
plt.xlabel('learning rate')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Hyperparameter.png')
plt.clf()

# Plot training & validation accuracy values
plt.plot(range(1, len(hist.history['accuracy'])+1), hist.history['accuracy'])
plt.plot(range(1, len(hist.history['accuracy'])+1), hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlim(0, len(hist.history['loss'])+1)
plt.ylim(0.7, 1)
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy.png')
plt.clf()

# Plot training & validation loss values
plt.plot(range(1, len(hist.history['accuracy'])+1), hist.history['loss'])
plt.plot(range(1, len(hist.history['accuracy'])+1), hist.history['val_loss'])
plt.title('Model loss')
plt.xlim(0, len(hist.history['loss'])+1)
plt.ylim(0.2, 0.8)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss.png')
plt.clf()


# ======================================================================================================================
# model evaluation
train_score, train_acc = model.evaluate(x_train, y_train,
                            batch_size=batch_size)
test_score, test_acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Train loss:{}, Train accuracy:{}, Test loss:{}, Test accuracy:{}'.format(train_score,
                                                                                train_acc,
                                                                                test_score,
                                                                                test_acc))