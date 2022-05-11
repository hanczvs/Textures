import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from keras import datasets, layers, models, losses, Model
from main import get_files
from main import dtdTest, dtdTrain, dtdValidation
from keras.models import load_model
from sklearn.metrics import classification_report
BATCH_SIZE = 100
IMAGE_SIZE = [224, 224]
NUM_CLASSES = 47

def labaelAccuracy():
    model = load_model('Mode2l.h5')
    print(model.summary())
    train_dataset, val_dataset, test_dataset = get_files()
    names=['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']
    y = np.concatenate([y for x, y in test_dataset], axis=0)
    Y_test=np.argmax(y,axis=1)
    y_pred = np.argmax(model.predict(test_dataset), axis=1)
    print(classification_report(Y_test, y_pred,target_names =names))

def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='SAME'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='SAME'),
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


def trainTest():
    train_dataset, val_dataset, test_dataset = get_files()

    base_model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False,
                                                         input_shape=(*IMAGE_SIZE, 3))

    base_model.trainable = False
    '''
    conv_kwargs = {
        "padding"             : "SAME",
        "activation"          : tf.keras.layers.LeakyReLU(alpha=0.2),
        "kernel_initializer"  : tf.random_normal_initializer(stddev=.1)
    }

    model = tf.keras.Sequential([
        base_model,tf.keras.layers.Conv2D(100, 3, strides=(2, 2), **conv_kwargs),
        tf.keras.layers.Conv2D(100, 3, strides=(2, 2), **conv_kwargs),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(47, activation='softmax')
    ])

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    '''
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.BatchNormalization(renorm=True),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(47, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics='accuracy')

    model.summary()

    early = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)

    history = model.fit(train_dataset, batch_size=100, epochs=20, validation_data=val_dataset, callbacks=[early])

    model.save("Model.h5")

    def plot(history):
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
        plt.savefig('figure.png')

    plot(history)
    print(model.evaluate(test_dataset))


if __name__ == "__main__":
    labaelAccuracy()
    #trainTest()