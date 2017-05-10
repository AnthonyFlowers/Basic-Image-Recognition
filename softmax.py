from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import data_helpers

# Dataset from https://www.cs.toronto.edu/~kriz/cifar.html
def main():

    ## Sets the time the program started ##
    begin_time = time.time()

    ## Parameter definitions ##
    batch_size = 100 # Amount of images in each batch to process
    learning_rate = 0.005
    max_steps = 1000 # Amount of steps the training operator is run

    ## Prepare/load data ##
    data_sets = data_helpers.load_data()

    ## Define input placeholders that specify the input datas type and shape ##
    # None allows us to use as many images for our tests in this tensor and 3072 sets the max amount of pixels as floats stored per image
    images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])
    # None allows this tensor to store as many image labels in this tensor as integers (0-9)
    labels_placeholder = tf.placeholder(tf.int64, shape=[None])

    ## Define variables: the values to optimize ##
    # A tensor with two parameters one to store the images pixel values and the other to store the weight values for each class/image type
    weights = tf.Variable(tf.zeros([3072, 10]))
    # Bias allows us to alwayse start with non-zero class score say if an image was all black
    biases = tf.Variable(tf.zeros([10]))


    ## Define the classifier's result ##
    # tf.matmul multiplies the images_placeholder matrix by the weights matrix
    # We add biases to keep the resulting values from equalling zero
    logits = tf.matmul(images_placeholder, weights) + biases

    ## Define the loss function ##
    # tf.nn.sparse_softmax... measures the probability error in discrete classification tasks where each class is mutually exclusive
    # tf.reduce_mean computes the mean of each element across the returned dimesions of the tensor
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder))

    ## Define training operation ##
    # This tensor applies the gradient descent optimizing algorithm
    # This also uses the .minimize() method to have the program focus on minimizing the loss function
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    ## Operation comparing prediction with true label ##
    # tf.argmax checks the largest vector between each logits value and 1
    # tf.equal checks if each argmax equals the labels_placeholder
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

    ## Operation calculating the accuracy of our predictioins ##
    # tf.cast casts correct_prediction values as a tf.float32
    # tf.reduce_mean takes the resulting vector and calculates the mean of each element across the returned dimensions of the tensor
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create saver to save progress
    saver = tf.train.Saver({'weights': weights, 'biases': biases})

    tf.add_to_collection('train_step', train_step)

    # Create tensorflow session
    sess = tf.Session()

    ### Running the graph ###

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Run training for max_steps times
    for i in range(max_steps):
        # Generate input data batch
        # Numpy takes random data sets in batch sizes of 'batch_size'
        indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
        # Takes the indices that numpy generated and pulls the batches from the images_train dictionary value from data_set
        images_batch = data_sets['images_train'][indices]
        # Takes the indices that numpy generated and pulls the batches from the labels_train dictionary value from data_set
        labels_batch = data_sets['labels_train'][indices]

        # Perform a single training step feeding the batch data into the train_step tensor
        sess.run(train_step, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})


        # Print current accuracy every 100 steps
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

            # Supposed to save current progress
            saver.save(sess, 'models/model')


    # Evaluate training with test set of data
    test_accuracy = sess.run(accuracy, feed_dict={images_placeholder: data_sets['images_test'], labels_placeholder: data_sets['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))

    # Print out time it took to train
    endTime = time.time()
    print('Total time: {:5.2f}s'.format(endTime - begin_time))

    # Test images against trained network
    # print(data_sets['labels_test'][0])
    # print(data_sets['images_test'][0])
    # print()
    # if sess.run(correct_prediction, feed_dict={images_placeholder: data_sets['images_test'], labels_placeholder: data_sets['labels_test']})[0]:
    ammount_correct = 0
    for x in range(10):
        # print(data_sets['classes'][data_sets['labels_test'][x]])
        if sess.run(correct_prediction, feed_dict={images_placeholder: data_sets['images_test'], labels_placeholder: data_sets['labels_test']})[x]:
            print(data_sets['classes'][data_sets['labels_test'][x]] + ': Correct Guess')
            ammount_correct += 1
        else:
            print(data_sets['classes'][data_sets['labels_test'][x]] + ': Incorrect Guess')
    print('Amount guessed correctly:', ammount_correct)





if __name__ == '__main__':
    main()
