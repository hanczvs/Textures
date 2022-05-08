import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

def label_encoder_pass(method, labels):
    if method == 'onehot':
        return one_hot(np.array(labels),47)
    elif method == 'elmo':
        return ElmoEmbeddingLayer(labels)
    elif method == 'bert':
        return
    elif method == 'lstm':
        return lstm(np.array(labels))
    else:
        raise NotImplementedError

def one_hot(labels, class_size):
    """
    Create one hot label matrix of size (N, C)

    Inputs:
    - labels: Labels Tensor of shape (N,) representing a ground-truth label
    for each MNIST image
    - class_size: Scalar representing of target classes our dataset 
    Returns:
    - targets: One-hot label matrix of (N, C), where targets[i, j] = 1 when 
    the ground truth label for image i is j, and targets[i, :j] & 
    targets[i, j + 1:] are equal to 0
    """
    targets = np.zeros((labels.shape[0], class_size))
    for i, label in enumerate(labels):
        targets[i, label] = 1
    targets = tf.convert_to_tensor(targets)
    targets = tf.cast(targets, tf.float32)
    return targets


def lstm(labels):
    """
    Create one hot label matrix of size (N, C)

    Inputs:
    - labels: Labels Tensor of shape (N,) representing a ground-truth label
    for each MNIST image
    Returns:
    - whole_seq_output: output
    """
    vocab_size = 47
    embedding_size = 128
    rnn_size = 256
    E = tf.Variable(tf.random.normal([vocab_size, embedding_size], stddev=.1, dtype=tf.float32))
    RNN = tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=True, activation= 'relu')
    embedding = tf.nn.embedding_lookup(E, labels)
    whole_seq_output, final_memory_state, final_carry_state = RNN(embedding, None)
    return whole_seq_output

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.load('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                            name="{}_module".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(
            K.squeeze(K.cast(x, tf.string), axis=1),
            as_dict=True,
            signature='default',
            )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)