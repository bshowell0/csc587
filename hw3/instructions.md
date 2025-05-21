# **Generative Models**

In this homework you will build and experiment with a convolutional GAN and a diffusion model for representation learning and image generation.

Your implementation should use the Keras module in Tensorflow 2 (import tensorflow.keras).

# **Part 1: GAN on Frey Dataset**

**1.1: Build the model**

Build and train a convolutional GAN on the Frey dataset, following the provided example notebook.

Your generator network should take in a latent vector and output an image. To do this, the first layer should be a dense layer with 7\*5\*128 outputs. Then reshape this output using a Reshape layer to size (7,5,128). Now you have converted your vector into a tensor of 128 7x5 images. Now you can use a Conv2DTranspose with stride 2 to upsample that into 256 14x10 images. Use one more Conv2DTranspose with stride 2 and 1 output channel to finally output a 28x20 image. Following the recommendations of DCGAN, use LeakyReLU activations with alpha=0.1 on all layers except the output layer.

The discriminator network should take an 28x20x1 image and output a probability value. Use Conv2D layers with stride 2 and ReLU activation. The first layer can have 256 channels and the second 128. Then flatten the tensor and do a final dense layer with a single output and sigmoid activation.

(Feel free to modify the network architecture and training settings as you like \-- this is not required, though.)

**1.2: Train the model**

Train the model following the MNIST example notebook.

Plot the discriminator and generator loss curves after training. Does the system seem to have converged?

**1.3: Evaluate the results**

Sample random images from `[-1 1]` box. Do the images appear similar to the training images? Do you see evidence of mode collapse in the results of GAN training? (Does the generator seem to be able to generate everything that is in the training set?)

# **Part 2: Latent Diffusion Model on Frey Dataset**

Build a latent diffusion model on the Frey dataset.  You will train a diffusion model to learn the distribution of the latent space of an autoencoder. To sample new images, you will first sample a latent vector using the diffusion model, and then decode it into an image.

**2.1: Build an autoencoder**

Train an MLP autoencoder on the Frey dataset. You can use the autoencoder you built in HW2. The bottleneck should be two-dimensional.

After training the autoencoder, extract and store the embedding vectors for the training images using encoder.predict().

**2.2: Build and train the reverse process model**

The reverse process model should take as input a two-dimensional embedding vector and the timestep `t` and output a two-dimensional vector. In-between should be a simple feedforward MLP. I used three hidden layers of width 128, but the design is up to you.


Use an Embedding layer to map the timestep to a vector the same size as your hidden layers. Then concatenate that vector to the two-dimensional input (along the channels axis) before processing with the MLP.

Train the model on the embedding vectors from part 2.1 using the custom training loop code provided in the diffusion example notebook.

**2.3: Evaluate the results**

Sample 1000 random 2D latent vectors using the diffusion sampling process (see the example notebook). Plot the training embedding vectors and your sampled 2D points on top. Did the diffusion model accurately learn the distribution of the latent space?

Then decode the first 10 sampled latent vectors into images. Do the images appear similar to the training images? Do you see evidence of mode collapse in the results of diffusion training? (Does the generator seem to be able to generate everything that is in the training set?)
