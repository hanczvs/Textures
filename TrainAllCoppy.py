import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape
from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Lambda, Activation, Resizing)
import random
import numpy as np
from main import getTest, getTrain


class Autoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        ## TODO: Implement call function
        return self.decoder(self.encoder(inputs))


class VAE(tf.keras.Model):
    def __init__(self, input_size, latent_size=25):
        super(VAE, self).__init__()
        self.encoder = Sequential()

        '''
        self.input_size = input_size # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = 400  # H_d

        self.encoder.add(Flatten())
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(tf.keras.layers.Dense(self.hidden_dim,activation='linear')) #encoder

        self.encoder.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
        self.encoder.add(Conv2D(16, 3, activation='relu'))
        self.encoder.add(MaxPooling2D(pool_size=2))
        self.encoder.add(Dropout(rate=0.25))
        self.encoder.add(Conv2D(32, 3, activation='relu'))
        self.encoder.add(Conv2D(64, 3, activation='relu'))
        self.encoder.add(MaxPooling2D(pool_size=2))
        self.encoder.add(Dropout(rate=0.25))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(units=128, activation='relu'))
        self.encoder.add(Dropout(rate=0.25))
        self.encoder.add(Dense(25, activation='softmax'))  # should we use a softmax here??

        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
                '''

        # self.decoder.add(tf.keras.layers.Reshape((1,300,400)))

    def fit(self, x, y, testx, testy):
        self.encoder.add(Resizing(224, 224, interpolation="bilinear", input_shape=x.shape[1:]))
        self.encoder.add(Conv2D(96, 11, strides=4, padding='same'))  # alexnet
        self.encoder.add(Lambda(tf.nn.local_response_normalization))
        self.encoder.add(Activation('relu'))
        self.encoder.add(MaxPooling2D(3, strides=2))
        self.encoder.add(Conv2D(256, 5, strides=4, padding='same'))
        self.encoder.add(Lambda(tf.nn.local_response_normalization))
        self.encoder.add(Activation('relu'))
        self.encoder.add(MaxPooling2D(3, strides=2))
        self.encoder.add(Conv2D(384, 3, strides=4, padding='same'))
        self.encoder.add(Activation('relu'))
        self.encoder.add(Conv2D(384, 3, strides=4, padding='same'))
        self.encoder.add(Activation('relu'))
        self.encoder.add(Conv2D(256, 3, strides=4, padding='same'))
        self.encoder.add(Activation('relu'))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(4096, activation='relu'))
        self.encoder.add(Dropout(0.5))
        self.encoder.add(Dense(4096, activation='relu'))
        self.encoder.add(Dropout(0.5))
        self.encoder.add(Dense(47, activation='softmax'))

        self.encoder.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy,
                             metrics=['accuracy'])
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = train_dataset.shuffle(buffer_size=24).batch(batch_size=64)
        history = self.encoder.fit(train_dataset, epochs=40)

        '''
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))
        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].title.set_text('Training Loss vs Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend(['Train', 'Val'])
        axs[1].plot(history.history['accuracy'])
        axs[1].plot(history.history['val_accuracy'])
        axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend(['Train', 'Val'])
        '''

        '''
        #self.encoder.compile(optimizer='Adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM), metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        self.encoder.compile(optimizer='Adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        #self.encoder.compile(optimizer='Adam',loss=tf.keras.losses.SparseCategoricalCrossentropy())
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = train_dataset.shuffle(buffer_size=80).batch(100)
        print(self.encoder.predict(x))
        print("predict")
        self.encoder.fit(train_dataset, epochs=2)
        res=self.encoder.predict(testx)
        print(res)
        test_dataset = tf.data.Dataset.from_tensor_slices((testx, testy))
        test_dataset = test_dataset.batch(4)
        result = self.encoder.evaluate(test_dataset)
        #dict(zip(self.encoder.metrics_names, result))
        print(result)
       '''


class CVAE(tf.keras.Model):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.num_classes = num_classes  # C
        self.hidden_dim = 400  # H_d

        self.flat = Sequential()
        self.flat.add(Flatten())
        self.encoder = Sequential()
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))

        self.linear = tf.keras.layers.Dense(self.latent_size, activation='linear')

        self.decoder = Sequential()
        self.decoder.add(tf.keras.Input(shape=(self.latent_size + self.num_classes)))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.input_size, activation='sigmoid'))
        self.decoder.add(tf.keras.layers.Reshape((47)))

    def call(self, x, c):
        xf = self.flat(x)
        enc = tf.concat((xf, c), axis=1)
        out = self.encoder(enc)
        cnnInput = self.linear(out)
        z = CNN(cnnInput)
        z = tf.cast(z, dtype=tf.float32)
        dec = tf.concat((z, c), axis=1)
        probs = self.decoder(dec)
        return probs


def CNN(x):  # this part will be sourced out to the CNNs
    # coppied some of this model from this link https://gist.github.com/JulieProst/8000610500a67fda4b76e07efe585552
    # make sure to note this in the final handin
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(25, activation='softmax'))  # should we use a softmax here??
    return model(x)


def loss(probabilities, labels):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities, axis=-1))


def train(model, train_inputs, train_labels):
    c = list(zip(train_inputs, train_labels))
    random.shuffle(c)
    trainInputs, trainLabels = zip(*c)
    trainInputs = train_inputs
    trainLabels = train_labels

    for i in range(int(len(trainLabels))):
        # for i in range(int(len(trainLabels) / model.batch_size)):
        with tf.GradientTape() as tape:
            trainInputs[i] = tf.reshape(trainInputs[i], (1, 300, 400, 3))
            # trainOutput1 = (model.call(trainInputs[i*model.batch_size:(i+1)*model.batch_size]))
            trainOutput1 = (model.call(trainInputs[i]))
            # Loss = model.loss(trainOutput1,trainLabels[i*model.batch_size:(i+1)*model.batch_size])
            Loss = loss(trainOutput1, trainLabels[i])
            if (i % 100 == 0):
                print(i)
        gradients = tape.gradient(Loss, model.trainable_variables)
        tf.keras.optimizers.Adam(learning_rate=.01).apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    acc = 0
    # a=int(len(test_labels) / model.batch_size)
    a = int(len(test_labels))
    for i in range(a):
        # correct_predictions = tf.equal(tf.argmax(model.call(test_inputs[i*model.batch_size:(i+1)*model.batch_size]), 1), tf.argmax(test_labels[i*model.batch_size:(i+1)*model.batch_size], 1))/a
        correct_predictions = tf.equal(tf.argmax(model.call(test_inputs[i]), 1), tf.argmax(test_labels[i], 1)) / a
        acc += tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return acc


def trainTest():
    x, y, z = getTrain()
    trainData = tf.convert_to_tensor(x)
    trainLab = tf.convert_to_tensor(y)
    print(len(y))
    print(tf.convert_to_tensor(x).shape)
    x, y, z = getTest()
    testData = tf.convert_to_tensor(x)
    testLab = tf.convert_to_tensor(y)
    # first need to reshape all elements in x so that theyre the same size (train has 332X436, and 300X400) (test has 510X413)
    conv_kwargs = {
        "padding": "SAME",
        "activation": tf.keras.layers.LeakyReLU(alpha=0.2),
        "kernel_initializer": tf.random_normal_initializer(stddev=.1)
    }

    ## TODO: Make encoder and decoder sub-models
    ae_model = Autoencoder(
        encoder=tf.keras.Sequential([tf.keras.layers.Conv2D(10, 3, strides=(2, 2), **conv_kwargs),
                                     tf.keras.layers.Conv2D(10, 3, strides=(2, 2), **conv_kwargs),
                                     tf.keras.layers.Conv2D(10, 3, strides=(2, 2), **conv_kwargs)], name='ae_encoder'),
        decoder=tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(10, 3, strides=(2, 2), **conv_kwargs),
            tf.keras.layers.Conv2DTranspose(10, 3, strides=(2, 2), **conv_kwargs),
            tf.keras.layers.Conv2DTranspose(1, 3, strides=(2, 2), **conv_kwargs)
        ], name='ae_decoder'), name='autoencoder')
    ae_model.build(input_shape=trainData.shape)  ## Required to see architecture summary
    initial_weights = ae_model.get_weights()  ## Just so we can reset out autoencoder

    #   ae_model.summary()
    #  ae_model.encoder.summary()
    # ae_model.decoder.summary()

    model = VAE(300 * 400)
    print("model fit")
    model.fit(trainData, trainLab, testData, testLab)
    print("fitted")
    # train(model,trainData, trainLab)
    # print(test(model,testData, testLab))


if __name__ == "__main__":
    trainTest()
