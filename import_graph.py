from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import data_helpers

def main():

    # Parameter definitions
    batch_size = 100
    learning_rate = 0.005
    max_steps = 1000

    # Prepare data
    data_sets = data_helpers.load_data()



    images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])
    labels_placeholder = tf.placeholder(tf.int64, shape=[None])

    # Define variables: the values to optimize
    weights = tf.Variable(tf.zeros([3072, 10]))
    biases = tf.Variable(tf.zeros([10]))


    # Define the classifier's result
    logits = tf.matmul(images_placeholder, weights) + biases

    # Define the loss function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder))

    # Define training operation
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Operation comparing prediction with true label
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

    # Operation calculating the accuracy of our predictioins
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('models/model.meta')
        new_saver.restore(sess, 'models/model')

        train_step = tf.get_collection('train_step')[0]

        # Init variables
        sess.run(tf.global_variables_initializer())

        # Generate input data batch
        indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
        images_batch = data_sets['images_train'][indices]
        labels_batch = data_sets['labels_train'][indices]

        for step in range(1000):
            # Print current accuracy every 100 steps
            if step % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})
                print('Step {:5d}: training accuracy {:g}'.format(step, train_accuracy))



            sess.run(train_step, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})
        # Evaluate training with test set of data
        test_accuracy = sess.run(accuracy, feed_dict={images_placeholder: data_sets['images_test'], labels_placeholder: data_sets['labels_test']})
        print('Test accuracy {:g}'.format(test_accuracy))





if __name__ == '__main__':
    main()
