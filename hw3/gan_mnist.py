# -*- coding: utf-8 -*-
"""gan_mnist.py

This notebook shows how to build a network similar to a DCGAN Deep Convolutional Generative Adversarial Network (DCGAN) for representation learning and image generation on the MNIST dataset.

"""

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.rc('image', cmap='gray')
from matplotlib import pyplot as plt

"""We grab the MNIST dataset and select just the zero digits."""

from tensorflow.keras.datasets import mnist

# x contains images, y contains integer labels (0-9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[y_train<1]
y_train = y_train[y_train<1]

x_test = x_test[y_test<1]
y_test = y_test[y_test<1]

print(x_train.shape)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_train[i])
    plt.axis('off')
plt.show()

x_train = (x_train.astype('float32')/255.)*2-1
x_test = (x_test.astype('float32')/255.)*2-1

"""# Model definition

Here we build a generator and discriminator network. The generator uses Conv2DTranspose to upsample images and LeakyReLU as the activation function, as recommended in the DCGAN paper. The discriminator uses strided convolutions instead of max pooling, also following the recommendations of DCGAN.
"""

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

def build_generator(latent_dim):
    inputs = Input((latent_dim,),name='generator_input')
    h1 = Dense(7*7*128,activation=None)(inputs)
    h1 = LeakyReLU(0.1)(h1)
    h1 = Reshape((7,7,128))(h1)
    h2 = Conv2DTranspose(256,3,strides=2,padding='same')(h1)
    h2 = LeakyReLU(0.1)(h2)
    output = Conv2DTranspose(1,3,strides=2,padding='same')(h2)
    return Model(inputs=inputs,outputs=output)

def build_discriminator():
    inputs = Input((28,28,1),name='encoder_input')
    h1 = Conv2D(256,3,strides=2,padding='same',activation='relu')(inputs)
    h2 = Conv2D(128,3,strides=2,padding='same',activation='relu')(h1)
    flat = Flatten()(h2)
    output = Dense(1,activation='sigmoid')(flat)
    return Model(inputs=inputs,outputs=output)

latent_dim = 2
generator = build_generator(latent_dim)
discriminator = build_discriminator()

print(generator.summary())
print(discriminator.summary())

"""# GAN Training

We have to manually set up a training loop because model.fit() doesn't support separate training of the generator and discriminator.
"""

g_opt = Adam(3e-4)
d_opt = Adam(3e-4)

num_iter = 5000
batch_size = 32

gen_loss_history = []
discrim_loss_history = []

# fixed set of random latent vectors to inspect development of generator output during training
latent_vectors = np.random.uniform(-1,1,size=(5,latent_dim))

# binary crossentropy loss function for discriminator training
loss_fn = tf.keras.losses.binary_crossentropy

for iter in range(num_iter):
    # select a random batch of training image indices
    inds = np.random.randint(len(x_train),size=batch_size)

    # get real data and labels
    x_real = x_train[inds]
    y_real = np.ones((batch_size,1))

    # get fake data (samples from generator) and labels
    x_fake = generator.predict(np.random.uniform(-1,1,size=(batch_size, latent_dim)), batch_size=batch_size, verbose=False)
    y_fake = np.zeros((batch_size,1))

    # compute discriminator loss -- label real images as 1 and fake images as 0
    with tf.GradientTape() as tape:
        real_loss = tf.reduce_mean(loss_fn(y_real,discriminator(x_real)))
        fake_loss = tf.reduce_mean(loss_fn(y_fake,discriminator(x_fake)))
        discrim_loss = 0.5*(real_loss + fake_loss)

    # compute gradients w.r.t. discriminator weights and update discriminator
    grads = tape.gradient(discrim_loss,discriminator.trainable_variables)
    d_opt.apply_gradients(zip(grads,discriminator.trainable_variables))

    # compute generator loss -- label fake images as 1
    with tf.GradientTape() as tape:
        x_gen = generator(np.random.uniform(-1,1,size=(batch_size, latent_dim)))
        gen_loss = tf.reduce_mean(loss_fn(y_real,discriminator(x_gen)))

    # generator updated
    grads = tape.gradient(gen_loss,generator.trainable_variables)
    g_opt.apply_gradients(zip(grads,generator.trainable_variables))

    # add losses to log
    gen_loss_history.append(gen_loss.numpy())
    discrim_loss_history.append(discrim_loss.numpy())

    # periodic summary output and generator sample visualization
    if iter % 100 == 0:
        print('iter %d: discriminator loss: %.2f\tgenerator loss: %.2f'%(iter+1,discrim_loss.numpy(),gen_loss.numpy()))
        x_gen = generator.predict(latent_vectors,verbose=False)
        for i in range(len(x_gen)):
            plt.subplot(1,len(x_gen),i+1)
            plt.imshow(np.squeeze(x_gen[i]))
        plt.show()

"""Plotting the loss curves"""

plt.plot(gen_loss_history)
plt.plot(discrim_loss_history)
plt.xlabel('iter')
plt.ylabel('loss')
plt.legend(['gen','discrim'])
plt.show()

"""## Generating new images

We generate images by sampling from a regularly spaced grid on [-1 1]x[-1 1].
"""

coords = np.linspace(-1,1,num=10)
x,y = np.meshgrid(coords,coords,indexing='xy')
embeddings = np.stack([x.flatten(),y.flatten()],axis=1)
plt.scatter(embeddings[:,0],embeddings[:,1])
plt.show()

plt.figure(figsize=(20, 20))
result = generator.predict(embeddings)
n = 0
for i in range(10):
    for j in range(10):
        plt.subplot(10,10,n+1)
        plt.imshow(np.squeeze(result[n]),cmap='gray')
        plt.axis('off')
        n = n + 1
plt.show()

