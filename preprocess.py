import tensorflow as tensorflow
import tensorflow_datasets as tfds

ds = tfds.load('dtd',split="train", shuffle_files=True)
print((ds.cardinality().numpy()))

for example in ds.take(1):
    image, label, file_name =  example["image"], example["label"], example["file_name"]
#https://github.com/sagerpascal/describable-textures-dataset/blob/master/dataset.py
