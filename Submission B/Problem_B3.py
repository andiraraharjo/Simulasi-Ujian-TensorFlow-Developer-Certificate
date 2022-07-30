# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator

def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()


    TRAINING_DIR = "data/rps/"
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.1)

    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        target_size=(150, 150),
                                                        batch_size=10,
                                                        class_mode='categorical',
                                                        subset='training')

    validation_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        target_size=(150, 150),
                                                        batch_size=10,
                                                        class_mode='categorical',
                                                        subset='validation')



    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_acc') > 0.83):
                print("\nReached 83.0% Validation accuracy so cancelling training!")
                self.model.stop_training = True

    callback = myCallback()
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])


    model.fit_generator(train_generator,
                        epochs=5,
                        verbose=1,
                        validation_data=validation_generator,
                        callbacks=[callback])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B3()
    model.save("model_B3.h5")

