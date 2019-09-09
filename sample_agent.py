from gym_torcs import TorcsEnv
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Flatten, Convolution2D, MaxPool2D, Reshape, Input, Concatenate
from keras.optimizers import Adam
from rl.core import Processor
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

random.seed(42)

run_num = 1

class TorcsProcessor(Processor):
    def process_observation(self, observation):
        focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, vision = observation

        vec_input = np.array(speedX, speedY, speedZ)
        img_input = vision


        return img_input, vec_input

class Agent(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action
        self.training = True
        self.init_model()

    def init_model(self):

        img_input = Input((4096,3))
        vec_input = Input((3,))

        x1 = Reshape((64, 64, 3), input_shape=(4096, 3))(img_input)
        x1 = Convolution2D(64, (3, 3), activation='relu')(x1)
        x1 = MaxPool2D()(x1)
        x1 = Convolution2D(32, (3, 3), activation='relu')(x1)
        x1 = MaxPool2D()(x1)
        x1 = Convolution2D(16, (3, 3), activation='relu')(x1)
        x1 = MaxPool2D()(x1)
        x1 = Flatten()(x1)
        x12 = Concatenate()([x1,vec_input])
        x12 = Dense(256, activation='relu')(x12)
        output = Dense(self.dim_action, activation='tanh')(x12)


        self.model = Model(inputs=[img_input,vec_input],outputs=[output])


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

processor = TorcsProcessor()

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}

dqn = DQNAgent(model=agent.model, processor=processor, nb_actions=agent.dim_action, memory=memory, nb_steps_warmup=50,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env,nb_steps=5,verbose=2,visualize=True)
