import tensorflow as tf
from keras import datasets, layers, models, losses, Model
from main import getTest,getTrain
# keras imports for the dataset and building our neural network
from main import getValidation
import matplotlib.pyplot as plt
from main import get_files
class VAE(tf.keras.Model):
    def __init__(self, input_size, latent_size=25):
        super(VAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = 400  # H_d

        """
        self.flat=Sequential()
        self.flat.add(Flatten())
        self.encoder = Sequential()
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))

        self.encoder.add(tf.keras.layers.Dense(self.latent_size, activation='linear'))
                """
        self.encoder = Sequential()
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
        self.encoder.add(Dense(47, activation='softmax'))  # should we use a softmax here??

        """
        self.decoder = Sequential()
        self.decoder.add(tf.keras.Input(shape=(self.latent_size+self.num_classes)))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.input_size, activation='sigmoid'))
        self.decoder.add(tf.keras.layers.Reshape((47)))
        """

        #self.decoder.add(tf.keras.layers.Reshape((1,300,400)))

    def call(self, x):
        return self.encoder(x)

class CVAE(tf.keras.Model):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.num_classes = num_classes # C
        self.hidden_dim = 400 # H_d

        """
        self.flat=Sequential()
        self.flat.add(Flatten())
        self.encoder = Sequential()
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))
        self.encoder.add(Dense(self.hidden_dim, activation='relu'))

        self.encoder.add(tf.keras.layers.Dense(self.latent_size, activation='linear'))
                """
        self.encoder = Sequential()
        self.encoder.add(Flatten())
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


        """
        self.decoder = Sequential()
        self.decoder.add(tf.keras.Input(shape=(self.latent_size+self.num_classes)))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.hidden_dim, activation='relu'))
        self.decoder.add(Dense(self.input_size, activation='sigmoid'))
        self.decoder.add(tf.keras.layers.Reshape((47)))
        """

    def call(self, x, c):
        xf=self.flat(x)
        enc=tf.concat((xf,c),axis=1)
        probs=self.encoder(enc)
        """
        cnnInput=self.linear(out)
        z=CNN(cnnInput)
        z=tf.cast(z,dtype=tf.float32)
        dec=tf.concat((z,c),axis=1)
        probs=self.decoder(dec)
        """
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

    for i in range(int(len(trainLabels))):
    #for i in range(int(len(trainLabels) / model.batch_size)):
        with tf.GradientTape() as tape:
            trainInputs[i]=tf.reshape(trainInputs[i],(1,300,400,3))
            #trainOutput1 = (model.call(trainInputs[i*model.batch_size:(i+1)*model.batch_size]))
            trainOutput1 = (model.call(trainInputs[i]))
            #Loss = model.loss(trainOutput1,trainLabels[i*model.batch_size:(i+1)*model.batch_size])
            Loss = loss(trainOutput1,trainLabels[i])
            if(i%100==0):
                print(i,Loss)
        gradients = tape.gradient(Loss, model.trainable_variables)
        tf.keras.optimizers.Adam(learning_rate=.01).apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    acc=0
    #a=int(len(test_labels) / model.batch_size)
    a=int(len(test_labels))
    for i in range(a):
        #correct_predictions = tf.equal(tf.argmax(model.call(test_inputs[i*model.batch_size:(i+1)*model.batch_size]), 1), tf.argmax(test_labels[i*model.batch_size:(i+1)*model.batch_size], 1))/a
        test_inputs[i] = tf.reshape(test_inputs[i], (1, 300, 400, 3))
        a1=int(tf.argmax(model.call(test_inputs[i]), 1).numpy()[0])
        a2=int(test_labels[i].numpy())
        print(a1,a2)
        correct_predictions = tf.equal(a1, a2)/a
        acc+=tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return acc

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

def testFunction():

    # loading the dataset
    (X_train, y_train,z)=getTrain()
    (X_test, y_test,z2) = getTest()

    # # building the input vector from the 32x32 pixels
    X_train = tf.reshape(X_train,(X_train.shape[0], 224, 224, 3))
    X_test = tf.reshape(X_test,(X_test.shape[0], 224, 224, 3))
    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 47
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)

    # building a linear stack of layers with the sequential model
    #model=VGGModel()
    #model(tf.keras.Input(shape=(224,224,3))
    out=tf.keras.applications.vgg16.VGG16(X_train,classes=47)

    model = Sequential()

    # convolutional layer
    model.add(Conv2D(50, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(400, 400, 3)))

    # convolutional layer
    model.add(Conv2D(75, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(125, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # flatten output of conv
    model.add(Flatten())

    # hidden layer
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.3))
    # output layer
    model.add(Dense(47, activation='softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the model for 10 epochs
    model.fit(x=z, batch_size=64, epochs=10, validation_data=z2)

def vggModel():
    print("init")
    x_train, y_train, z = getTrain()
    x_test, y_test, z = getTest()
    x_val, y_val, z = getValidation()
    model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
    for layer in model.layers:
        layer.trainable = False

    x = layers.Flatten()(model.output)
    x = layers.Dense(1000, activation='relu')(x)
    predictions = layers.Dense(47, activation = 'softmax')(x)

    head_model = Model(inputs=model.input, outputs=predictions)
    head_model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_dataset = train_dataset.shuffle(buffer_size=24).batch(64)
    test_dataset = test_dataset.batch(64)
    val_dataset = val_dataset.batch(64)


    history = head_model.fit(train_dataset, batch_size=64, epochs=75, validation_data=val_dataset)


def trainTest2():
    x_train, y_train, z = getTrain()
    x_test, y_test, z = getTest()
    x_val, y_val, z = getValidation()

    def conv_block(filters):
        block = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
            tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D()
        ]
        )

        return block

    def dense_block(units, dropout_rate):
        block = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        return block

    def build_model():
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(224, 224, 3)),

            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(),

            conv_block(32),
            conv_block(64),

            conv_block(128),
            tf.keras.layers.Dropout(0.2),

            conv_block(256),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Flatten(),
            dense_block(512, 0.7),
            dense_block(128, 0.5),
            dense_block(64, 0.3),

            tf.keras.layers.Dense(47, activation='softmax')
        ])

        return model

    def build_model():
        base_model = tf.keras.applications.VGG16(input_shape=(224,224, 3),
                                                 include_top=False,
                                                 weights='imagenet')

        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(47, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics='accuracy')

        return model
    model = build_model()
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("texture_model.h5",
                                                           save_best_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                             restore_best_weights=True)

    def exponential_decay(lr0, s):
        def exponential_decay_fn(epoch):
            return lr0 * 0.1 ** (epoch / s)

        return exponential_decay_fn

    exponential_decay_fn = exponential_decay(0.01, 20)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    history = model.fit(
            train_dataset, epochs=20,
            validation_data=val_dataset,
            callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler]
        )


    print(model.evaluate(test_dataset))

    """
    
    base_model = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(47, activation = 'softmax')(x)

    head_model = Model(inputs=base_model.input, outputs=predictions)
    head_model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))


    train_dataset = train_dataset.shuffle(buffer_size=64).batch(64)
    test_dataset = test_dataset.batch(64)
    val_dataset = val_dataset.batch(64)



    history = head_model.fit(train_dataset, batch_size=64, epochs=5, validation_data=val_dataset)


    def plot():
        fig, axs = plt.subplots(2, 1, figsize=(15,15))
        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].title.set_text('Training Loss vs Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend(['Train','Val'])
        axs[1].plot(history.history['accuracy'])
        axs[1].plot(history.history['val_accuracy'])
        axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend(['Train', 'Val'])
    plot()
    print(head_model.evaluate(test_dataset))
"""
def trainTest3():

    train_dataset,val_dataset,test_dataset=get_files()
    base_model = tf.keras.applications.VGG16(input_shape=(224,224, 3), include_top=False,weights='imagenet')
    base_model.trainable = False

    def dense_block(units, dropout_rate):
        block = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        return block

    def conv_block(filters):
        block = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
            tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.MaxPool2D()
        ]
        )

        return block
    x = layers.Flatten(base_model.output)
    x=layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x=layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    #x=layers.MaxPool2D()(x)
    x=conv_block(32)(x)
    x=conv_block(64)(x)
    x=conv_block(128)(x)
    x=layers.Dropout(.2)(x)
    x=conv_block(256)(x)
    x=layers.Dropout(.2)(x)
    x=layers.Flatten()(x)
    x=dense_block(512, 0.7)(x)
    x=dense_block(128, 0.5)(x)
    x=dense_block(64, 0.3)(x)

    predictions = layers.Dense(47, activation = 'softmax')(x)

    head_model = Model(inputs=base_model.input, outputs=predictions)
    head_model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    history = head_model.fit(train_dataset, batch_size=16, epochs=5, validation_data=val_dataset)


    def plot():
        fig, axs = plt.subplots(2, 1, figsize=(15,15))
        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].title.set_text('Training Loss vs Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend(['Train','Val'])
        axs[1].plot(history.history['accuracy'])
        axs[1].plot(history.history['val_accuracy'])
        axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend(['Train', 'Val'])
    plot()
    print(head_model.evaluate(test_dataset))


if __name__ == "__main__":
    trainTest3()
