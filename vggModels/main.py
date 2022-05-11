# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

NUM_CLASSES = 47
BATCH_SIZE = 32

def get_files():
    train_ds, val_ds, test_ds = tfds.load('dtd', split=['train', 'validation', 'test'], shuffle_files=True, batch_size = BATCH_SIZE)

    train_ds = train_ds.map(lambda items: (items["image"]/255, tf.one_hot(items["label"], NUM_CLASSES)))

    val_ds = val_ds.map(lambda items: (items["image"]/255, tf.one_hot(items["label"], NUM_CLASSES)))

    test_ds = test_ds.map(lambda items: (items["image"]/255, tf.one_hot(items["label"], NUM_CLASSES)))

    def convert(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    def pad(image, label):
        image,label = convert(image, label)
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)
        return image, label

    train_ds = (train_ds.map(pad))

    val_ds = (val_ds.map(pad))

    test_ds = (test_ds.map(pad))

    return train_ds, val_ds, test_ds