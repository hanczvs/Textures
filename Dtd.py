import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html"
_DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"

class Dtd2(tfds.core.GeneratorBasedBuilder):
  """Describable Textures Dataset (DTD)."""

  VERSION = tfds.core.Version("3.0.1")

  def _info(self):
    names_file = tfds.core.tfds_path(
        os.path.join("image_classification", "dtd_key_attributes.txt"))
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            "file_name": tfds.features.Text(),
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(names_file=names_file),
        }),
        homepage=_URL)

  def _split_generators(self, dl_manager):
    data_path = dl_manager.download_and_extract('https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz')
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(data_path=data_path, split_name="train1")),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(data_path=data_path, split_name="test1")),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs=dict(data_path=data_path, split_name="val1")),
    ]

  def _generate_examples(self, data_path, split_name):
    with tf.io.gfile.GFile(
        os.path.join(data_path, "dtd", "labels", split_name + ".txt"),
        "r") as split_file:
      for line in split_file:
        fname = line.strip()
        label = os.path.split(fname)[0]
        record = {
            "file_name": fname,
            "image": os.path.join(data_path, "dtd", "images", fname),
            "label": label,
        }
        yield fname, record