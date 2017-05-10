# Basic Image Recognition

## Synopsis

This project is a demonstration using [TensorFlow](https://www.tensorflow.org/) (https://www.tensorflow.org/) and the CIFAR-10 image [data set](http://www.cs.toronto.edu/~kriz/cifar.html) (http://www.cs.toronto.edu/~kriz/cifar.html) to train a neural network to classify images based on pixel rgb occurrences at each position and calculating a weight to base a prediction on.

## Code Example

Running the softmax.py module will run the main program which iterates over the dataset randomly to learn/improve correct classification probability.

Example:

`python softmax.py`
```
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
```


## Motivation

I chose to do this project to get my foot in the door with machine learning and neural networks in general. I focused on understanding the basic fundamentals of machine learning through neural networks.

## Installation

For this project I used a virtual environment that uses Python 3.5.2 and then install each dependency.

[Ubuntu Install](https://www.tensorflow.org/install/install_linux)
[Mac OS X Install](https://www.tensorflow.org/install/install_mac)
[Windows Install](https://www.tensorflow.org/install/install_windows)

This project includes a requirements.txt file to install the dependencies.

This project also requires the [Microsoft Visual C++ 2015 Redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=53587) on Windows (and probably other platforms too).

The dataset I used is from [Here](http://www.cs.toronto.edu/~kriz/cifar.html).

After downloading and extracting the data set the cifar-10-batches-py folder can be moved to the projects main directory.

[Learning Multiple Layers of Features from Tiny Images](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.

## API Reference

The TensorFlow documentation can be found [Here](https://www.tensorflow.org/api_docs/python/): https://www.tensorflow.org/api_docs/python/
A very helpful tutorial that I found explained how each Tensor or TensorFlow method worked can be found [Here](http://www.wolfib.com/Image-Recognition-Intro-Part-1/): http://www.wolfib.com/Image-Recognition-Intro-Part-1/

## Tests

There are no tests included with this project.

Some tests that could be written include:
* Testing how the program weights different classes, say compare the weights on the dog class compared to the cat class because they may share similar weights on certain pixel areas.
* Use TensorBoard to show the weights of pixels value per class.

## Contributors

Let people know how they can dive into the project, include important links to things like issue trackers, irc, twitter accounts if applicable.

If you would like to help work on this project or find any bugs/changes that you could improve on, do send a message on [Twitter](https://twitter.com/alteiar).


## License

A short snippet describing the license (MIT, Apache, etc.)
