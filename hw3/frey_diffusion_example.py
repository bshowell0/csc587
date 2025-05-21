# -*- coding: utf-8 -*-
"""frey_diffusion_example.ipynb

# Diffusion example

This notebook provides a simple implementation of a [Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239.pdf) or DDPM (Ho et al., 2020).

The DDPM is a variational method, where introduce latent variables to help in modeling $p(\mathbf{x})$. Here we call the data $\mathbf{x}_0$ and we introduce a series of latent variables $\mathbf{x}_1,\ldots,\mathbf{x}_{T}$, each having the same size as $\mathbf{x}_0$.

In the *forward process*, we slowly add noise to $\mathbf{x}_0$ according to a fixed variance schedule $\beta_t$:

$$q(\mathbf{x}_t|\mathbf{x}_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})$$.

In other words, to sample $\mathbf{x}_t$, we add a normally-distributed noise vector $\mathbf{\epsilon_t}$ to $\mathbf{x}_{t-1}$. In the *reverse process*, we attempt to remove this noise to recover $\mathbf{x}_{t-1}$ from $\mathbf{x}_t$.
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import math
from tqdm.auto import tqdm, trange
from tensorflow.keras.utils import get_file
from scipy.io import loadmat
from matplotlib import pyplot as plt

"""We will experiment with the Frey dataset which contains 28 x 20 grayscale images."""

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train,axis=-1)
    x_train = x_train.astype('float32')
    x_train /= 255
    x_train = x_train * 2 - 1
    return x_train

def get_frey_data():
    path = get_file('frey_rawface.mat','https://www.dropbox.com/scl/fi/m70sh4ef39pvy01czc63r/frey_rawface.mat?rlkey=5v6meiap55z68ada2roxwxuql&dl=1')
    data = np.transpose(loadmat(path)['ff'])
    x_train = np.reshape(data,(-1,28,20,1))
    x_train = x_train.astype('float32')
    x_train /= 255
    x_train = x_train * 2 - 1
    return x_train

num_hidden = 128
num_layers = 12

T = 1000
betas = np.linspace(1e-4,.02,T)
sigmas = np.sqrt(betas)
alphas = 1 - betas
alphas_cumprod = np.cumprod(alphas,axis=-1)

batch_size = 128

"""In our example, the reverse process model is a convolutional neural network (CNN) which takes as input $\mathbf{x}_t$ and timestep $t$, and outputs its estimate of $\mathbf{\epsilon}_t$, the noise added at timestep $t$ in the forward process.

To condition the network on $t$, we use an [Embedding layer](https://keras.io/api/layers/core_layers/embedding/). The Embedding layer stores a list of T learnable vectors accessible by an index (the timestep). Each vector will have size $H \cdot W$ so that it can be resized into an $H \times W$ image and concatenated to the input image along the channels axis.
"""

def build_reverse_process_model(H,W,num_layers,num_hidden):
    """ Builds the reverse process model.

    Arguments:
        H: image height
        W: image width
        num_layers: number of layers in CNN
        num_hidden: width of each CNN layer

    Returns:
        Keras model
    """
    # create image and timestep inputs
    image_input = layers.Input((H,W,1))
    timestep_input = layers.Input((1,))

    # create embedding layer with T vectors of size H*W
    embedding = layers.Embedding(T,H*W,embeddings_initializer='glorot_normal')

    # look up embedding vectors for timesteps
    conditional = embedding(timestep_input)

    # reshape embedding vector into an image
    conditional = keras.layers.Reshape((H,W,1))(conditional)

    # concatenate to input along channels axis
    x = keras.layers.Concatenate()([image_input,conditional])

    # process in convolutional layers
    for i in range(num_layers):
        x = layers.Conv2D(num_hidden,3,activation='relu',padding='same',use_bias=True)(x)

    # output is estimate of noise
    x = layers.Conv2D(1,3,activation=None,padding='same')(x)

    model = keras.Model(inputs=[image_input,timestep_input],outputs=x)
    return model

"""The sampling procedure starts by sampling $\mathbf{x}_T$ as random noise from a unit normal distribution.    Then at each timestep from $T-1$ to $0$ it subtracts the noise estimate from the reverse process model.    The actual update formula is a bit more complicated and comes from the Ho et al. paper."""

def sample(model,shape):
    """ Samples from the diffusion model.

    Arguments:
        model: reverse process model
        shape: shape of data to be sampled; should be [N,H,W,1]
    Returns:
        Sampled images
    """
    # sample normally-distributed random noise (x_T)
    x = np.random.normal(size=shape)

    # iterate through timesteps from T-1 to 0
    for t in trange(T-1,-1,-1):
        # sample noise unless at final step (which is deterministic)
        z = np.random.normal(size=shape) if t > 0 else np.zeros(shape)

        # estimate correction using model conditioned on timestep
        eps = model.predict([x,np.ones((shape[0],1))*t],verbose=False)

        # apply update formula
        sigma = sigmas[t]
        a = alphas[t]
        a_bar = alphas_cumprod[t]
        x = 1/np.sqrt(a)*(x - (1-a)/np.sqrt(1-a_bar)*eps)+sigma*z
    return x

"""Now we get the data and build the model."""

x_train = get_frey_data()
H,W = x_train.shape[1:3]

model = build_reverse_process_model(H,W,num_layers,num_hidden)
model.summary()

"""At each training iteration, we do the following:
* Sample a random batch of images and timesteps
* Apply the forward process to obtain the noised version of each image at the corresponding timestep
* Estimate the noise using the reverse process model
* Compute mean squared error between the estimated and actual noise
"""

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

opt = keras.optimizers.Adam(3e-4)
loss_fn = keras.losses.MeanSquaredError()

epochs = 100
for epoch in range(epochs):
    total_loss = 0

    pbar = tqdm(total=len(train_dataset))
    for step, x_batch_train in enumerate(train_dataset):
        t = np.random.randint(T,size=len(x_batch_train))

        with tf.GradientTape() as tape:
            noise = np.random.normal(size=x_batch_train.shape)

            at = alphas_cumprod[t]
            at = np.reshape(at,(-1,1,1,1))
            inputs = np.sqrt(at) * x_batch_train + (1-at)*noise
            est_noise = model([tf.convert_to_tensor(inputs),tf.convert_to_tensor(t)])

            loss_value = loss_fn(noise, est_noise)

            total_loss += loss_value.numpy()

        grads = tape.gradient(loss_value, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

        pbar.update(1)
    pbar.close()

    total_loss /= len(train_dataset)
    print(f'loss at epoch {epoch}: {total_loss}')

"""Once training has completed we can sample a batch of images and inspect the results."""

image_sample = sample(model,x_train[:10].shape)

fig,axes = plt.subplots(1,len(image_sample))
for i in range(len(image_sample)):
    axes[i].imshow(image_sample[i])
fig.show()

"""As yuo can see, the results are far from perfect, probably because of the limitations of our simple CNN model design. As you will see in the homework, a latent diffusion model can produce much better results even when using a simple neural network model."""

