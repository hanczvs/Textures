import numpy as np
import tensorflow as tf
from Resnet101 import ResnetEncoder
import random
from main import getTrain,getTest

class Model1(tf.keras.Model):
    def __init__(self, class_num, backbone='resnet101', pretrained_backbone=True, use_feats=(4,),
                 fc_dims=(512,)):
        super(Model1, self).__init__()
        self.img_encoder = ResnetEncoder(backbone,pretrained_backbone, use_feats)

        in_dim = self.img_encoder.out_dim
        fc_layers = []
        if len(fc_dims) > 0:
            for fc_i, fc_dim in enumerate(fc_dims):
                fc_layer = tf.Sequential(tf.Linear(in_dim, fc_dim),
                                         tf.BatchNormalization(fc_dim),
                                         tf.ReLU())
                fc_layers.append(fc_layer)
                in_dim = fc_dims
        fc_layers.append(tf.Linear(in_dim, class_num))
        self.fc_layers = tf.Sequential(*fc_layers)

    def forward(self, x):
        img_feats = self.img_encoder(x)
        pred_scores = self.fc_layers(img_feats)
        return pred_scores

def loss(self, probabilities, labels):
        # We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #cross = tf.losses.BinaryCrossEntropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        #loss = tf.reduce_mean(cross(targets, pred) * class_weights)
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities, axis=-1)) #this is what we used before, one above is the one they used.

def train(model, train_inputs, train_labels):
    c = list(zip(train_inputs, train_labels))
    random.shuffle(c)
    trainInputs, trainLabels = zip(*c)
    trainInputs=train_inputs
    trainLabels=train_labels

    for i in range(int(len(train_inputs))):
        with tf.GradientTape() as tape:
            trainOutput1,tup = (model.call(trainInputs[i],initial_state = None))
            loss = model.loss(trainOutput1,trainLabels[i])
        gradients = tape.gradient(loss, model.trainable_variables)
        tf.keras.optimizers.Adam(learning_rate=.01).apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    trainInputs=test_inputs
    trainLabels=test_inputs
    acc=0
    a=int(len(trainInputs))
    for i in range(int(len(trainInputs))):
        trainOutput1,tup=model.call(trainInputs[i],initial_state = None)
        acc+=model.loss(trainOutput1, trainLabels[i])/a
    return np.exp(acc)

def trainTest():
    x, y, z = getTrain()
    trainData = x
    trainLab = y
    x, y, z = getTest()
    testData = x
    testLab = y
    model = Model1(1)
    train(model,trainData, trainLab)
    print(model.test(testData, testLab))

if __name__ == "__main__":
    trainTest()

    pass
