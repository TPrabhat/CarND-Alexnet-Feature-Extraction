import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import pandas as pd

# Load traffic signs data.
with open("train.p", mode='rb') as f:
    train = pickle.load(f)

nb_classes = np.shape(pd.read_csv("signnames.csv"))[0]


X, y = train['features'], train['labels']

# Split data into Training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#print(np.shape(y_train), np.shape(y_test))

# Define placeholder and resize operation.

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
resized = tf.image.resize_images(x, (227,227))

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
# determine shape of the second last layer of AlexNet

shape=(fc7.get_shape().as_list()[-1], nb_classes)
fc8_W = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
fc8_b = tf.Variable(tf.zeros(nb_classes))

logits = tf.nn.xw_plus_b(fc7, fc8_W, fc8_b)
probs = tf.nn.softmax(logits)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: Train and evaluate the feature extraction model.
