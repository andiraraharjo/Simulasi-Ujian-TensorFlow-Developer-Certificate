# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# ===========================================================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_A4():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_data = imdb['train']
    test_data = imdb['test']

    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []

    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    # YOUR CODE HERE

    training_label_seq = np.array(training_labels)
    test_label_seq = np.array(testing_labels)

    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    oov_tok = "<OOV>"
    
    tokenizer =  Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(training_sentences)
    train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_length,truncating=trunc_type)

    test_sequences = tokenizer.texts_to_sequences(testing_sentences)
    test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length)


    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    train_padded = np.array(train_padded)
    test_padded = np.array(test_padded)

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if ((logs.get('val_acc') > 0.83) and (logs.get('acc') > 0.83)):
                print("\nReached 93.0% Validation accuracy so cancelling training!")
                self.model.stop_training = True

    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    callback = myCallback()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(train_padded, training_label_seq, epochs=10,
              validation_data=(test_padded, test_label_seq), verbose=2,callbacks=[callback])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A4()
    model.save("model_A4.h5")

