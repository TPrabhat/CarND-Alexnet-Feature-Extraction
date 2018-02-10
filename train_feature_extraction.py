import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import pandas as pd
from sklearn.utils import shuffle


# Load traffic signs data.
with open("train.p", mode='rb') as f:
    train = pickle.load(f)

nb_classes = np.shape(pd.read_csv("signnames.csv"))[0]


X, y = train['features'], train['labels']

# Split data into Training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
num_examples = len(X_train)

#print(np.shape(X_train))

# Define placeholder and resize operation.

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int64, None)
keep_prob = tf.placeholder(tf.float32)
x_resized = tf.image.resize_images(x, (227,227))


# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x_resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
# determine shape of the second last layer of AlexNet

shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8_W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
fc8_b = tf.Variable(tf.zeros(nb_classes))

logits = tf.nn.xw_plus_b(fc7, fc8_W, fc8_b)

probs = tf.nn.softmax(logits)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

one_hot_y = tf.one_hot(y_train, len(X_train))
rate = 0.001
EPOCHS = 30
BATCH_SIZE = 128

# Define loss, training, accuracy operations.

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8_W, fc8_b])

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)

        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})

        validation_accuracy = evaluate(X_test, y_test)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")