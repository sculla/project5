from gym_torcs import TorcsEnv
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPool2D, Reshape
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

random.seed(42)

run_num = 1


class Agent(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action
        self.training = True
        self.init_model()

    def init_model(self):
        self.model = Sequential()
        self.model.add(Reshape((64, 64, 3), input_shape=(4096, 3)))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPool2D())
        self.model.add(Convolution2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPool2D())
        self.model.add(Convolution2D(16, (3, 3), activation='relu'))
        self.model.add(MaxPool2D())
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.dim_action, activation='tanh'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def save_weights(self):
        self.model.save_weights(f'output/weights/torcs/best_run.h5f')

    def load_weights(self):
        self.model.load_weights(f'output/weights/torcs/best_run.h5f')


vision = True
episode_count = 10
max_steps = 100
reward = 0
done = False
step = 0

# Generate a Torcs environment
env = TorcsEnv(vision=vision, throttle=False)

agent = Agent(dim_action=1)

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}

dqn = DQNAgent(model=agent.model, nb_actions=agent.dim_action, memory=memory, nb_steps_warmup=50,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env,nb_steps=5,verbose=2,visualize=True)
