import numpy as np
import gym
# import tensorflow as tf


from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPool2D
from keras.optimizers import Adam
from keras.layers import Permute, Reshape
from keras.backend import permute_dimensions, reshape

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def car_model_2d():
    nb_actions = env.action_space.shape[0]

    WINDOW_LENGTH = 2
    INPUT_SHAPE = env.observation_space.shape

    model = Sequential()
    model.add(Convolution2D(160, (3, 3), activation='relu', input_shape=(320, 240, 3)))
    model.add(MaxPool2D())
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Convolution2D(16, (3, 3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='tanh'))

    print(model.summary())
    return model


#
# K.tensorflow_backend._get_available_gpus()
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#

ENV_NAME = 'CarRacing-v0'



# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
state_shape = env.observation_space.shape
np.random.seed(123)
env.seed(123)
print(env.action_space)

nb_actions = env.action_space.shape[0]

WINDOW_LENGTH = 1
INPUT_SHAPE = env.observation_space.shape
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
# width, height, channels, window
# model.add(Permute((2, 3, 4, 1), input_shape=input_shape))
# # time steps into one 2d convo
model.add(Reshape((96, 96, (3 * WINDOW_LENGTH)),input_shape=input_shape))
model.add(Convolution2D(16, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env,nb_steps=1,verbose=2,visualize=False)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# import random
# random.seed(123)
# n_iter = 5
# life_memory = []
#
# tot_reward = 0
# for _game_idx in range(n_iter):
#     old_observation = env.reset()  # this starts a new "episode" and returns the initial observation
#     done = False
#
#     episode_mem = []
#     while not done:
#         new_action = env.action_space.sample()
#         observation, reward, done, info = env.step(new_action)
#         tot_reward += reward
#         episode_mem.append(dict(zip(['observation', 'new_action', 'reward', 'game_idx'],
#                                     (old_observation, new_action, reward, _game_idx))))
#         old_observation = observation
#     n_steps = len(episode_mem)
#     for step_idx, epi_mem in enumerate(episode_mem):
#         epi_mem['tot_reward'] = tot_reward
#         epi_mem['decay_reward'] = step_idx * tot_reward / n_steps
#     life_memory.extend(episode_mem)
#
# g_df = pd.DataFrame(life_memory)
# g_df.to_pickle('data/initial_games.pkl')

# After training is done, we save the final weights.
dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)