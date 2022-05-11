import tensorflow as tf
import matplotlib.pyplot as plt
from keras import datasets, layers, models, losses, Model
from main import get_files
from main import dtdTest, dtdTrain, dtdValidation

BATCH_SIZE = 32
IMAGE_SIZE = [224, 224]
NUM_CLASSES = 47


def trainTest():
    x_val,y_val,z_val=dtdValidation()
    print("15")
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    print("17")
    val_dataset = val_dataset.batch(64)
    print("19")
    x_test,y_test,z_test=dtdTest()
    print("20")
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    print("21")
    test_dataset = test_dataset.batch(64)
    print("23")
    x_train,y_train,z_train=dtdTrain()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=64).batch(64)

    base_model = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))

    base_model.trainable = False

    conv_kwargs = {
        "padding": "SAME",
        "activation": tf.keras.layers.LeakyReLU(alpha=0.2),
        "kernel_initializer": tf.random_normal_initializer(stddev=.1)
    }

    model = tf.keras.Sequential([
        base_model, tf.keras.layers.Conv2D(100, 3, strides=(2, 2), **conv_kwargs),
        tf.keras.layers.Conv2D(100, 3, strides=(2, 2), **conv_kwargs),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(47, activation='softmax')
    ])
    '''
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    '''
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics='accuracy')

    history = model.fit(train_dataset, batch_size=100, epochs=20, validation_data=val_dataset)

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
    trainTest()