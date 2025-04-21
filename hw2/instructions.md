# **Homework 2: Unsupervised Representation Learning**

100 points
Due date: Sunday, May 4, 11:59pm

In this homework you will explore how autoencoders can generate useful representations of data in an unsupervised manner.  You will train autoencoders on the MNIST and Frey datasets, visualize what filters the network learns, and test the quality of the learned representation for data compression, generation, and classification.

Your implementation should use the Keras module in Tensorflow 2 (import tensorflow.keras).

# **Part 1: Frey face compression and generation**

## **1.1 Autoencoder setup and training**

##

Load the Frey dataset and convert the images to have `float32` type and a range of `[-1 1]`. Display some of the images.

Set up an autoencoder, following the example in the linear autoencoder notebook.  Unlike the linear autoencoder, your autoencoder should use a multi-layer perceptron for the encoder and decoder and have a two-dimensional latent space.  The hidden layers in your encoder and decoder should have some non-linear activation such as ReLU or Leaky ReLU, except the embedding or bottleneck layer (last layer of encoder) which should have a tanh activation.  The last layer of the decoder should have size `28*20` and should have linear (None) activation.

The exact design of the encoder and decoder is up to you.  For reference, I used a two-layer MLP for the encoder and decoder, and the number of channels was `64-32-2-32-64`.

## **1.2 Analysis and Visualization**

Test the ability of the autoencoder to compress and decompress the images.  Compare some input images to their reconstructions after running the autoencoder.  What effect does the autoencoder have on the images?  How does it compare to the linear autoencoder?

Visualize the output of the encoder (run on the training data) as a scatter plot.  Give some observations about the output.  Does it seem to be using all of the possible output space?

Generate new faces by decoding a set of embedding points on `[-1 1] x [-1 1]` grid (see `np.meshgrid`).  Give some observations about the resulting images.

Test interpolation between two images (see the example from the linear autoencoder notebook).  How do the interpolations compare to the linear autoencoder?

# **Part II: MNIST digit classification with unsupervised pre-training**

Set up and train a similar MLP autoencoder on the MNIST dataset.  Use only digits 0 and 1 to make the task a little easier.  Use mean squared error loss for this dataset--I found it to work more reliably than mean absolute error.

After training the autoencoder, obtain the embedding vectors of all training and testing images.

Then, create and train another network that will classify the embedding vectors produced by your encoder.  Train the network on the training data and test it (model.evaluate()) on the testing data.  What test accuracy are you able to achieve?  Discuss the effectiveness of unsupervised pre-training in this experiment.
