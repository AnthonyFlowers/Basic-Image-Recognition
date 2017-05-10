# Basic-Image-Recognition

## Synopsis

This project is a demonstration using TensorFlow (https://www.tensorflow.org/) and the CIFAR-10 image data set (http://www.cs.toronto.edu/~kriz/cifar.html) to train a neural network to classify images.

## Code Example

Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.
Running the softmax.py module will run the main program which iterates over the dataset randomly to learn/improve correct classification probability.

Ex:
'''
python softmax.py
'''
Step     0: training accuracy 0.45
Step   100: training accuracy 0.25
Step   200: training accuracy 0.29
Step   300: training accuracy 0.42
Step   400: training accuracy 0.49
Step   500: training accuracy 0.31
Step   600: training accuracy 0.52
Step   700: training accuracy 0.34
Step   800: training accuracy 0.38
Step   900: training accuracy 0.41
Test accuracy 0.2586
Total time:  9.68s
cat: Correct Guess
ship: Incorrect Guess
ship: Incorrect Guess
plane: Incorrect Guess
frog: Incorrect Guess
frog: Incorrect Guess
car: Incorrect Guess
frog: Incorrect Guess
cat: Incorrect Guess
car: Correct Guess
Amount guessed correctly: 2


## Motivation

I chose to do this project to get my foot in the door for machine learning and neural networks in general.

## Installation

This project includes a requirements.txt file to install the dependencies.

The dataset I used is from [here](http://www.cs.toronto.edu/~kriz/cifar.html)
[Learning Multiple Layers of Features from Tiny Images](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests

Describe and show how to run the tests with code examples.

## Contributors

Let people know how they can dive into the project, include important links to things like issue trackers, irc, twitter accounts if applicable.

## License

A short snippet describing the license (MIT, Apache, etc.)
