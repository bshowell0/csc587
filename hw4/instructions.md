# **Homework 4: Reinforcement Learning**

In this homework you will build and experiment with deep Q-learning for reinforcement learning on a couple classic RL problems.

Your implementation should use the Keras module in Tensorflow (import tensorflow.keras) and Gymnasium (import gymnasium as gym).

# **Part 1: Table-based Q-learning (Frozen Lake)**

In this part you will write code to learn a Q-table for the Frozen Lake (FrozenLake-v1) environment.  Note that you can do this part entirely in Numpy without Keras / Tensorflow. 

**To replicate my results exactly, set the random seeds to 1234\.**

The [frozen lake environment](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) consists of a 4x4 map:

SFFF  
FHFH  
FFFH  
HFFG

Where S is the start, G is the goal, F is a free space and H is a hole.  When you create the environment, pass the argument “is\_slippery=False” so that the environment is deterministic and you will always move in the desired direction.

The state is an integer from 0 to 15 indicating the current grid cell.  The sixteen grid cells are numbered 0 to 15 in row-major order with 0 being the start and 15 being the goal.

There are four possible actions: left (0), down (1), right (2), and up (3).

A reward of \+1 is given for reaching the goal state; all other states have zero reward.

Here is pseudo-code for how to learn the Q-table using Bellman updates:

1. Initialize a 16x4 matrix of zeros to represent the Q-table Q(*s*,*a*).  
2. Reset the environment (s \= env.reset()) to get the initial state *s*.  
3. Repeat for num\_step steps:  
4.     Sample an action *a* using an *epsilon-greedy* strategy  
5.     Apply the action to receive a new state *s’* and a reward *r*.  
6.     Calculate the expected cumulative payoff *y* as   
   *y \= r*                                              if the game is over or  
   *y* \= *r* \+ *gamma* \* max*a* Q(*s’*,*a*)     otherwise  
7.     Update Q using Q(*s*,*a*) \= Q(*s*,*a*) \+ lr\*(*y*\-Q(*s*,*a*))  
8.     If the game is over, reset the environment (s \= env.reset()); otherwise, update the current state with *s* \= *s’*.

For the epsilon-greedy strategy, I used epsilon=exp(-decay\_rate \* step) with decay\_rate=0.0001.  For the other parameters I used gamma \= 0.99, num\_steps=2000, and lr=.01.

Now test the policy learned by the Q-table on the environment as follows:

1. Reset the environment to get initial state *s*.  
2. Choose the next action *a* as the action with maximum utility: max*a* Q(*s*,*a*)  
3. Apply the action *a* to get the next state *s’*.  
4. Render the environment (env.render())  
5. Set *s* \= *s’.*  
6. Repeat 2-5 until done, or a maximum number of steps is reached.

I set the maximum number of steps to 10,000 – if this limit is reached, then the trial is marked as failed.  If you have properly learned the Q-table, you should reach the goal state every time.

Now learn a Q-table for the slippery frozen lake (is\_slippery=True).   How often does the resulting policy succeed over 1000 trials?  Now learn the Q-table again, but use 200,000 iterations and a decay\_rate of 0.00001.  Your success rate should have improved \-- why?

# 

# **Part 2: n-step Deep Q-Learning (Cart Pole)**

In this part you will implement n-step deep Q-learning to solve the CartPole-v1 environment.  Since we will be training a neural network to learn the Q-function, you will need to use Keras and Tensorflow in this part.

The [cart pole environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) consists of a pole on a cart which can move left and right.  The goal is to keep the pole from falling down.

The state consists of four continuous values: the cart position, cart velocity, pole angle, and pole angular velocity.

There are two possible actions: left (0) and right (1).

A reward of \+1 is given for each step taken, and the game ends when the pole angle \> 12 degrees, the absolute cart position \> 2.4, or the episode length is greater than 500 steps.

Your goal is to learn a policy which can keep the pole aloft for an average of at least 200 steps over 100 trials.

Your Q-function approximator should be a multi-layer perceptron with a single hidden layer with ReLU activation.  The input to the network is the state, and the output is a Q-value for each action.  Thus the input vector size is the size of the state (four) the output vector size is the number of actions (two).

Our approach here is to run through a complete episode of the environment before updating the Q-network.  Starting from the initialization state, you will play through the game using an epsilon-greedy policy until the “done” flag is raised.  Along the way you will accumulate lists for the previous state, action taken, reward received, and next state.  At the end of the episode, you calculate the discounted cumulative reward at each state.  Finally, you will use the action taken and discounted cumulative reward received at each step to calculate a batch of losses and update the neural network.  This process is repeated for some number of episodes.

Here is pseudo-code to implement n-step Q-learning:

1. Create a neural network to represent the Q-function Q(*s*,*a*).  
2. Initialize step counter to 1\.  
3. Repeat for num\_episodes episodes:  
4.     Reset the environment (s \= env.reset()) to get the initial state *s*.  
5.     Create empty lists for *s*, *a*, and *r*.  
6.     While not done:  
7.         Sample an action *a* using an *epsilon-greedy* strategy.  
8.         Apply the action to receive a new state *s’* and a reward *r*.  
9.         Append *s*, *a*, and *r* to their lists.  
10.         Set *s* \= *s’*  
11.         Increment the step counter.  
12.     Let *n* be the number of steps in the episode.  
13.     Calculate the cumulative payoff *y* at each step as:  
    *y*(*n*) *\= r*(*n*)  
    *y*(*i*) \= *r*(*i*) \+ *gamma* \* *y*(i+1)     for *i* \= *n*\-1,*n*\-2,...,1  
14.     Divide each *y*(*i*) by the maximum possible cumulative reward .  (This normalization helps stabilize training.  Note that the maximum possible per-step reward is *R*max \= 1.)  
15.     For each action *a*(*i*), calculate the loss *L(i) \=* (*y*(*i*) \- *Q*(*s*(*i*),*a*(*i*)))2  
16.     Calculate the average loss over all *L(i)*.  
17.     Calculate the gradient of the average loss and update the model.

For the epsilon-greedy strategy, I used epsilon=exp(-decay\_rate \* step) with decay\_rate=0.00001, where step is the global step count.  For the other parameters I used gamma \= 0.99, num\_episodes \= 3000, and lr=.001 with the Adam optimizer.  I used a hidden layer size of 32 in the Q-network.

**Note: Make sure that you do not include the calculation of y(i) in the calculation of the gradient.  You should not turn on GradientTape until after you have calculated the y(i) values.**

It is helpful to print the number of steps in the episode at each iteration and the current epsilon value to monitor the progress during training.

Once the network is trained, calculate how often the learned policy succeeds (reaches \>200 steps) over 100 episodes, and the average number of steps per episode.  

What happens if you increase the decay\_rate to 0.001 and set num\_episodes \= 100?  Why?