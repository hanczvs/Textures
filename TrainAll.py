import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers import (Conv2D,Dense,Dropout,Flatten,MaxPooling2D)
import random
import numpy as np
from main import getTest,getTrain

class VAE(tf.keras.Model):
    def __init__(self, input_size, latent_size=25):
        super(VAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = 400  # H_d
        self.encoder = Sequential()
        self.encoder.add(Flatten())
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(tf.keras.layers.Reshape((20,20,1)))

        self.linear = tf.keras.layers.Dense(self.latent_size,activation='linear')

        self.decoder = Sequential()
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(47, activation='sigmoid'))
        #self.decoder.add(tf.keras.layers.Reshape((1,300,400)))

    def call(self, x):

        out = self.encoder(x)
        cnnInput = self.linear(out)
        cnnOutput = CNN(cnnInput)
        pred = self.decoder(cnnOutput)
        return pred

class CVAE(tf.keras.Model):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.num_classes = num_classes # C
        self.hidden_dim = 400 # H_d

        self.flat=Sequential()
        self.flat.add(Flatten())
        self.encoder = Sequential()
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))

        self.linear = tf.keras.layers.Dense(self.latent_size, activation='linear')

        self.decoder = Sequential()
        self.decoder.add(tf.keras.Input(shape=(self.latent_size+self.num_classes)))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.input_size, activation='sigmoid'))
        self.decoder.add(tf.keras.layers.Reshape((47)))


    def call(self, x, c):
        xf=self.flat(x)
        enc=tf.concat((xf,c),axis=1)
        out=self.encoder(enc)
        cnnInput=self.linear(out)
        z=CNN(cnnInput)
        z=tf.cast(z,dtype=tf.float32)
        dec=tf.concat((z,c),axis=1)
        probs=self.decoder(dec)
        return probs


def CNN(x):# this part will be sourced out to the CNNs
#coppied some of this model from this link https://gist.github.com/JulieProst/8000610500a67fda4b76e07efe585552
#make sure to note this in the final handin
    model = Sequential()
    model.add(Conv2D(filters=16,kernel_size=3,activation='relu'))
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
    model.add(Dense(25, activation='softmax')) #should we use a softmax here??
    return model(x)

def loss(probabilities,labels):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities, axis=-1))

def train(model, train_inputs, train_labels):
    c = list(zip(train_inputs, train_labels))
    random.shuffle(c)
    trainInputs, trainLabels = zip(*c)
    trainInputs=train_inputs
    trainLabels=train_labels

    for i in range(int(len(train_inputs))):
        with tf.GradientTape() as tape:
            trainInputs[i]=tf.reshape(trainInputs[i],(1,300,400,3))
            trainOutput1 = (model.call(trainInputs[i]))
            Loss = loss(trainOutput1,trainLabels[i])
            if(i%100==0):
                print(i)
        gradients = tape.gradient(Loss, model.trainable_variables)
        tf.keras.optimizers.Adam(learning_rate=.01).apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    trainInputs=test_inputs
    trainLabels=test_labels
    acc=0
    a=int(len(trainInputs))
    for i in range(int(len(trainInputs))):
        trainInputs[i] = tf.reshape(trainInputs[i], (1, 300, 400, 3))
        trainOutput1=model.call(trainInputs[i])
        acc+=loss(trainOutput1, trainLabels[i])/a
    return np.exp(acc)

def trainTest():
    x, y, z = getTrain()
    trainData = x
    trainLab = y
    print(len(y))
    print(x[0].get_shape().as_list())
    x, y, z = getTest()
    testData = x
    testLab = y
    #first need to reshape all elements in x so that theyre the same size (train has 332X436, and 300X400) (test has 510X413)
    print(x[0].get_shape().as_list())
    model = VAE(300 * 400)
    train(model,trainData, trainLab)
    print(test(model,testData, testLab))

if __name__ == "__main__":
    trainTest()
