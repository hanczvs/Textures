# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from Dtd import Dtd2

NUM_CLASSES = 47
BATCH_SIZE = 1

def getTrain():
    dataset=tfds.load('dtd',split="train",shuffle_files=True)
    images=[tf.image.resize(example["image"], [224, 224]) for example in dataset]
    labels=[example["label"] for example in dataset]
    filenames=[example["file_name"] for example in dataset]
    return images,labels,filenames

def getValidation():
    dataset=tfds.load('dtd',split="validation",shuffle_files=True)
    images=[tf.image.resize(example["image"], [224, 224]) for example in dataset]
    labels=[example["label"] for example in dataset]
    filenames=[example["file_name"] for example in dataset]
    return images,labels,filenames

def getTest():
    dataset=tfds.load('dtd',split="test",shuffle_files=True)
    images=[tf.image.resize(example["image"], [224, 224]) for example in dataset]
    labels=[example["label"] for example in dataset]
    filenames=[example["file_name"] for example in dataset]
    return images,labels,filenames

def get_files():
    train_ds, val_ds, test_ds = tfds.load('dtd', split=['train', 'validation', 'test'], shuffle_files=False, batch_size = BATCH_SIZE)

    train_ds = train_ds.map(lambda items: (items["image"], tf.one_hot(items["label"], NUM_CLASSES)))

    val_ds = val_ds.map(lambda items: (items["image"], tf.one_hot(items["label"], NUM_CLASSES)))

    test_ds = test_ds.map(lambda items: (items["image"], tf.one_hot(items["label"], NUM_CLASSES)))

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

def dtdTrain(n):
    dt=Dtd2()
    dm = tfds.download.DownloadManager(download_dir='DLFinal')
    #print(labels)
    datapath='DLFinal\\extracted\\TAR_GZ.robot.ox.ac.uk_vgg_dtd_downl_dtd-r1.0.15ChVpSpJUKO1lhKDRgKqJTkUdVyVsM_56tbQc5X44gU.tar.gz'
    images, labels, filenames=[],[],[]
    for i in range(1,n+1):
        x= dt._generate_examples(datapath,'train'+str(i))
        labels=labels+[tf.keras.preprocessing.text.one_hot((example[1]["label"]),47) for example in x]
        x= dt._generate_examples(datapath,'train'+str(i))
        images=images+[tf.image.resize(tf.image.decode_image(tf.io.read_file(example[1]["image"])), [224, 224])/ 255.0 for example in x]
        x= dt._generate_examples(datapath,'train'+str(i))
        filenames=filenames+[example[1]["file_name"] for example in x]
        print(i)
    label = [tf.one_hot(i[0],47) for i in labels]
    return images,label,filenames
def dtdTest(n):
    dt=Dtd2()
    dm = tfds.download.DownloadManager(download_dir='DLFinal')
    #print(labels)
    datapath='DLFinal\\extracted\\TAR_GZ.robot.ox.ac.uk_vgg_dtd_downl_dtd-r1.0.15ChVpSpJUKO1lhKDRgKqJTkUdVyVsM_56tbQc5X44gU.tar.gz'
    images, labels, filenames=[],[],[]
    for i in range(1,n+1):
        x= dt._generate_examples(datapath,'test'+str(i))
        labels=labels+[tf.keras.preprocessing.text.one_hot((example[1]["label"]),47) for example in x]
        x= dt._generate_examples(datapath,'test'+str(i))
        images=images+[tf.image.resize(tf.image.decode_image(tf.io.read_file(example[1]["image"])), [224, 224])/ 255.0 for example in x]
        x= dt._generate_examples(datapath,'test'+str(i))
        filenames=filenames+[example[1]["file_name"] for example in x]
        print(i)
    label = [tf.one_hot(i[0],47) for i in labels]
    return images,label,filenames

def dtdValidation(n):
    dt=Dtd2()
    dm = tfds.download.DownloadManager(download_dir='DLFinal')
    #print(labels)
    datapath='DLFinal\\extracted\\TAR_GZ.robot.ox.ac.uk_vgg_dtd_downl_dtd-r1.0.15ChVpSpJUKO1lhKDRgKqJTkUdVyVsM_56tbQc5X44gU.tar.gz'
    images, labels, filenames=[],[],[]
    for i in range(1,n+1):
        x= dt._generate_examples(datapath,'val'+str(i))
        labels=labels+[tf.keras.preprocessing.text.one_hot((example[1]["label"]),47) for example in x]
        x= dt._generate_examples(datapath,'val'+str(i))
        images=images+[tf.image.resize(tf.image.decode_image(tf.io.read_file(example[1]["image"])), [224, 224])/ 255.0 for example in x]
        x= dt._generate_examples(datapath,'val'+str(i))
        filenames=filenames+[example[1]["file_name"] for example in x]
        print(i)
    label = [tf.one_hot(i[0],47) for i in labels]
    return images,label,filenames

if __name__ == "__main__":
    images,labels,filenames=dtdTrain()
    images,labels,filenames=dtdTest()
    images,labels,filenames=dtdValidation()

    print(len(images))
