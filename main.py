# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def getTrain():
    dataset=tfds.load('dtd',split="train",shuffle_files=True)
    images=[example["image"] for example in dataset]
    labels=[example["label"] for example in dataset]
    filenames=[example["file_name"] for example in dataset]
    return images,labels,filenames

def getValidation():
    dataset=tfds.load('dtd',split="validation",shuffle_files=True)
    images=[example["image"] for example in dataset]
    labels=[example["label"] for example in dataset]
    filenames=[example["file_name"] for example in dataset]
    return images,labels,filenames

def getTest():
    dataset=tfds.load('dtd',split="test",shuffle_files=True)
    images=[example["image"] for example in dataset]
    labels=[example["label"] for example in dataset]
    filenames=[example["file_name"] for example in dataset]
    return images,labels,filenames

if __name__ == '__main__':
    test=getTest()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
