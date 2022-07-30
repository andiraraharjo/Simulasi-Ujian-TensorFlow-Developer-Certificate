# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    labelencoder = LabelEncoder()
    bbc['category'] = labelencoder.fit_transform(bbc['category'])

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    x_train, x_test, y_train, y_test = train_test_split(bbc["text"], bbc["category"], test_size= 1 - training_portion)
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(x_train)
    train_sequences = tokenizer.texts_to_sequences(x_train)
    train_padded = pad_sequences(train_sequences, padding="post", maxlen=max_length)

    testing_sequences = tokenizer.texts_to_sequences(x_test)
    testing_padded = pad_sequences(testing_sequences, padding='post', maxlen=max_length)

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if ((logs.get('val_acc') > 0.91) and (logs.get('acc') > 0.91)):
                print("\nReached 91.0% Validation accuracy so cancelling training!")
                self.model.stop_training = True


    callback = myCallback()
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(train_padded, y_train, epochs=500,
              validation_data=(testing_padded, y_test), verbose=1,callbacks=[callback])
    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B4()
    model.save("model_B4.h5")
