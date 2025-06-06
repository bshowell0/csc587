# %% [markdown]
# ## Gymnasium examples
#
# Gymnasium is a Python module that provides many classic example environments for reinforcement learning tasks.  This notebook shows how to create and run an environment.

# %%
import gymnasium as gym

# %%
from matplotlib import pyplot as plt
from ipywidgets import interact, IntSlider

# %%
def show_video(images,captions):
  """ Show a sequence of images as an interactive plot. """
  def f(i):
    plt.imshow(images[i])
    plt.title(captions[i])
  interact(f, i=IntSlider(min=0, max=len(images)-1, step=1, value=0))

# %% [markdown]
# ### Frozen Lake
#
# The [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment consists of a 4x4 map:
#
#     SFFF
#     FHFH
#     FFFH
#     HFFG
#
# Where S is the start, G is the goal, F is a free space and H is a hole.  When you create the environment, pass the argument “is_slippery=False” so that the environment is deterministic and you will always move in the desired direction.
#
# The sixteen states are numbered 0 to 15 in row-major order with 0 being the start and 15 being the goal.
#
# There are four possible actions: left (0), down (1), right (2), and up (3).
#
# A reward of +1 is given for reaching the goal state; all other states have zero reward.
#

# %%
# Configuration parameters for the whole setup
seed = 1234
env = gym.make("FrozenLake-v1",is_slippery=False,render_mode='rgb_array')
env.reset(seed=seed)
env.action_space.seed(seed)

# %%
# reset the environment back to the initial state
state, _ = env.reset()

renders = []
captions = []

# get the initial state
renders.append(env.render())
captions.append(f'state: {state} action: reward: done:')

while True:
  # sample a random action
  action = env.action_space.sample()

  # apply the action to get the next state, reward, and done flag
  state, reward, done, _, _ = env.step(action)

  renders.append(env.render())
  captions.append(f'state: {state} action:{action} reward:{reward} done:{done}')

  if done:
      break
show_video(renders,captions)

# %% [markdown]
# ### Cart Pole environment
#
# The [CartPole-v1 environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) consists of a pole on a cart which can move left and right.  The goal is to keep the pole from falling down.
#
# The state consists of four continuous values: the cart position, cart velocity, pole angle, and pole angular velocity.
#
# There are two possible actions: left (0) and right (1).
#
# A reward of +1 is given for each step taken, and the game ends when the pole angle > 12 degrees, the absolute cart position > 2.4, or the episode length is greater than 500 steps.

# %%
seed = 1234
env = gym.make("CartPole-v1",render_mode='rgb_array')
env.reset(seed=seed)
env.action_space.seed(seed)

# %%
# reset the environment back to the initial state
state, _ = env.reset(seed=seed)

renders = []
captions = []

# get the initial state
renders.append(env.render())
captions.append(f'state: {state} action: reward: done:')

while True:
  # sample a random action
  action = env.action_space.sample()

  # apply the action to get the next state, reward, and done flag
  state, reward, done, _, _ = env.step(action)

  renders.append(env.render())
  captions.append(f'state: {state} action:{action} reward:{reward} done:{done}')

  if done:
      break
show_video(renders,captions)

# %%



