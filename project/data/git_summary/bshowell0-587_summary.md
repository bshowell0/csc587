### **Overall Summary**

This repository, `bshowell0-587`, is a comprehensive collection of coursework for a deep learning class, likely numbered 587. It showcases a series of practical implementations of core machine learning concepts, progressing from foundational neural networks to advanced generative and reinforcement learning models. The work is organized into four major homework assignments and a detailed project proposal, all implemented using TensorFlow 2 and Keras.

The assignments cover a wide range of topics:
- **Homework 1** serves as an introduction to TensorFlow and Keras, starting with basic tutorials and culminating in the construction of a Convolutional Neural Network (CNN) for MNIST digit classification. A key feature of this assignment is a detailed analysis of the model's robustness to image rotation, including a comparison between a standard model and one trained with rotation-based data augmentation.
- **Homework 2** delves into unsupervised representation learning using autoencoders. It features two parts: first, building a multi-layer perceptron (MLP) autoencoder to compress, reconstruct, and generate images from the Frey face dataset, with in-depth analysis of the 2D latent space. The second part demonstrates the power of unsupervised pre-training by using an autoencoder to learn embeddings for MNIST digits 0 and 1, and then training a highly efficient and accurate classifier on these low-dimensional embeddings.
- **Homework 3** focuses on generative models. The student builds and trains two different models on the Frey face dataset: a Deep Convolutional Generative Adversarial Network (DCGAN) and a more advanced Latent Diffusion Model. The Latent Diffusion implementation is particularly noteworthy, as it first uses an autoencoder to learn a 2D latent space and then trains a diffusion model to generate new samples within that latent space, which are subsequently decoded into images.
- **Homework 4** covers reinforcement learning. It begins with a from-scratch implementation of table-based Q-learning in NumPy to solve the Frozen Lake environment, analyzing the impact of stochasticity (`is_slippery`) and the need for extended training. The second part implements n-step Deep Q-Learning (DQL) with a Keras-based Q-network to master the CartPole environment, demonstrating the trade-offs between exploration and exploitation through different hyperparameter settings.

Finally, the `project/` directory contains a detailed proposal for a final project: building a personalized, on-device AI agent. The proposal outlines a plan to fine-tune the Qwen3-0.6B small language model with personal data and integrate it into the Gosling Android agent framework, demonstrating a strong grasp of current trends in on-device AI, model personalization (PEFT), and agentic systems. Overall, the repository is a strong portfolio of well-documented, practical deep learning projects.

### **Key Code and Structure Details**

#### **Homework 1: CNNs and Rotational Invariance (`hw1/main.ipynb`)**

This assignment goes beyond standard MNIST classification by analyzing the model's behavior under rotation.
- **2D Embedding Layer**: The CNN architecture includes an intermediate 2-dimensional `Dense` layer with a linear activation, which is explicitly used to visualize how the network learns to separate digit classes in a 2D space. This provides insight into the model's internal representations.
  ```python
  model = models.Sequential([
      # ... Conv layers ...
      layers.Dense(200, activation='relu', name='dense1'),
      layers.Dense(2, activation=None, name='embedding'), # 2D embedding layer
      layers.Dense(10, activation='softmax', name='output')
  ])
  ```
- **Rotation Analysis**: A systematic analysis is performed where test images are rotated in 10-degree increments from 0 to 360 degrees. For each rotation, the model's confidence (max probability) and predicted class are recorded and plotted. This reveals the model's brittleness to transformations not seen during training. For example, a '9' is shown to be misclassified as a '6' when rotated 180 degrees.
- **Data Augmentation for Robustness**: A second model is created with a `layers.RandomRotation` layer added at the beginning of the `Sequential` model.
  ```python
  augmented_model = models.Sequential([
      layers.Input(shape=(28, 28, 1)),
      layers.RandomRotation(factor=0.2, fill_mode='constant', fill_value=-1.0),
      # ... Same CNN architecture as before ...
  ])
  ```
  After retraining with this augmentation layer, the rotation analysis is repeated. The resulting plots demonstrate that the augmented model maintains much higher confidence in the correct class across a wider range of rotation angles, providing a clear, empirical validation of the benefits of data augmentation.

#### **Homework 2: Unsupervised Pre-training with Autoencoders (`hw2/main.ipynb`)**

This assignment effectively demonstrates two key applications of autoencoders.
- **Part 1: Frey Face Generation**: An MLP autoencoder with a 2D bottleneck is trained on the Frey face dataset. The notebook includes excellent visualizations:
    - **Latent Space Scatter Plot**: Shows that the learned embeddings form a distinct, curved manifold within the `[-1, 1]` space, indicating a structured representation.
    - **Grid-based Generation**: New faces are generated by decoding points from a regular grid in the latent space. This visualization clearly shows how different regions of the latent space correspond to different facial poses and expressions (e.g., smiling faces in one area, looking left/right along an axis). It also highlights how areas outside the learned manifold produce distorted, non-face-like images.
- **Part 2: Classification via Embeddings**: This section showcases a powerful transfer learning technique.
    1. An autoencoder with a 16-dimensional latent space is trained on MNIST digits 0 and 1 in an unsupervised manner (using only the images `x_train_01` as both input and target).
    2. The trained `mnist_encoder` is used as a feature extractor to convert the high-dimensional pixel data (784 dimensions) into compact 16-dimensional embeddings for both training and test sets.
    3. A very simple, lightweight classifier is then trained *exclusively* on these 16D embeddings.
    ```python
    classifier = Sequential([
        Input(shape=(mnist_latent_dim,)), # Input is the 16D embedding
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Binary classification for 0 vs 1
    ])
    ```
    This approach achieves nearly perfect accuracy (99.91%), demonstrating that unsupervised pre-training can effectively learn meaningful, low-dimensional features that make subsequent supervised tasks much easier and more efficient.

#### **Homework 3: Generative Models (GAN and Latent Diffusion) (`hw3/main.ipynb`)**

This assignment implements and compares two sophisticated generative models.
- **Part 1: Convolutional GAN**: A DCGAN is implemented to generate Frey faces. The training loop is written manually with `tf.GradientTape` to alternate between training the discriminator and the generator, which is standard practice for GANs. The generated images show some success but also classic GAN artifacts, with the notebook noting that "the nose to mouth area struggles to have defining features".
- **Part 2: Latent Diffusion Model**: This is a more advanced, two-stage process.
    1. **Autoencoder Pre-training**: An MLP autoencoder is first trained on the Frey dataset to compress images into a 2D latent space. The `encoder` and `decoder` models are saved.
    2. **Diffusion on Latent Space**: A diffusion model (DDPM) is then trained not on the images themselves, but on the 2D `latent_embeddings_train` generated by the autoencoder. The diffusion model is an MLP that takes a noisy 2D latent vector and a timestep `t` as input and predicts the noise that was added. Timestep information is fed to the model via an `Embedding` layer.
    ```python
    # Model takes a latent vector and a timestep
    latent_input = layers.Input(shape=(latent_dim,))
    timestep_input = layers.Input(shape=(1,))
    time_embed = layers.Embedding(input_dim=T_diffusion, output_dim=time_embedding_dim)(timestep_input)
    # ... MLP processes concatenated latent_input and time_embed ...
    ```
    3. **Sampling Process**: To generate a new image, the process is: (1) use the trained diffusion model to sample a new 2D latent vector, (2) feed this vector into the autoencoder's `decoder` to produce a full-size image. This approach is more stable than training a diffusion model directly on pixels and effectively learns the distribution of the latent manifold.

#### **Homework 4: Reinforcement Learning (`hw4/main.ipynb`)**

This assignment implements two core RL algorithms.
- **Part 1: Table-based Q-Learning**: This section provides a clean, from-scratch implementation of Q-learning for the Frozen Lake environment. The Bellman update is clearly implemented in the training loop. The key takeaway is the comparison between the deterministic and stochastic versions of the environment, showing that the agent's success rate on the slippery lake jumps from 0% to 71.8% when the number of training steps is increased 100-fold (from 2,000 to 200,000) and the epsilon decay rate is slowed. This empirically demonstrates the importance of sufficient exploration in stochastic environments.
- **Part 2: N-step Deep Q-Learning**: This section implements DQL for the continuous state space of CartPole. The approach is notable for its use of full-episode rollouts for updates.
    - **Episodic Updates**: The agent plays an entire episode to completion, storing all states, actions, and rewards in lists.
    - **N-step Return Calculation**: After the episode ends, the discounted cumulative reward (return) is calculated for each step by iterating *backwards* through the episode's rewards. This is a form of Monte Carlo update.
      ```python
      # running_return is y(i) from the instructions
      for i in reversed(range(n_steps)):
          running_return = episode_rewards[i] + gamma * running_return
          returns[i] = running_return
      ```
    - **Batch Network Update**: The entire episode's worth of `(state, action, calculated_return)` tuples is used to update the Q-network in a single batch gradient descent step. This is more stable than single-step updates. The notebook effectively contrasts a successful training run with a failed one, correctly identifying premature exploitation and insufficient data as the causes of failure in the latter.
