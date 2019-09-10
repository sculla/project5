from gym_torcs import TorcsEnv
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Flatten, Convolution2D, MaxPool2D, Reshape, Input, Concatenate, Permute
from keras.optimizers import Adam
from rl.core import Processor
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor
from PIL import Image
from keras.utils import plot_model

random.seed(42)
INPUT_SHAPE = (64,64)
WINDOW_LENGTH = 2
run_num = 1
vision = True

class TorcsProcessor(MultiInputProcessor):

    def __init__(self, nb_inputs):
        self.nb_inputs = nb_inputs

    def process_observation(self, observation):

        focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, vision = observation

        vec_input = np.array([[speedX, speedY, speedZ]])

        #image processing
        assert vision.shape == (4096,3)
        vision = vision.reshape((64,64,3))
        img = Image.fromarray(vision)
        img = img.resize(INPUT_SHAPE).convert('L') #grayscale
        img_input = np.array(img)
        assert img_input.shape == INPUT_SHAPE
        print(vec_input)
        # assert vec_input.shape == (3,)
        return (img_input / 255) #, speedY

class Agent(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action
        self.training = True
        self.init_model()

    def init_model(self):

        img_input = Input(((WINDOW_LENGTH,) + INPUT_SHAPE))
        # vec_input = Input((WINDOW_LENGTH,1,))
        x1 = Permute((3,2,1))(img_input)
        x1 = Convolution2D(64, (3, 3), activation='relu')(x1)
        x1 = MaxPool2D()(x1)
        x1 = Convolution2D(32, (3, 3), activation='relu')(x1)
        x1 = MaxPool2D()(x1)
        x1 = Convolution2D(16, (3, 3), activation='relu')(x1)
        x1 = MaxPool2D()(x1)
        # x12 = Concatenate(axis=-1)([x1,vec_input])
        x12 = Flatten()(x1)
        # x2 = Permute((2,1))(vec_input)
        # x2 = Flatten()(x2)
        # x12 = Concatenate(axis=-1)([x1,x2])
        # x12 = Concatenate(axis=-1)([x1,x2])
        x12 = Dense(256, activation='relu')(x12)
        output = Dense(self.dim_action, activation='tanh')(x12)
        # self.model = Model(inputs=[img_input,vec_input],outputs=[output])
        self.model = Model(inputs=[img_input],outputs=[output])
        plot_model(self.model, to_file='model.png')
        print(self.model.summary())

    def save_weights(self):
        self.model.save_weights(f'output/weights/torcs/best_run.h5f')

    def load_weights(self):
        self.model.load_weights(f'output/weights/torcs/best_run.h5f')





# Generate a Torcs environment
env = TorcsEnv(vision=vision, throttle=False)

agent = Agent(dim_action=1)

processor = TorcsProcessor(nb_inputs=1)

memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}

dqn = DQNAgent(model=agent.model, processor=processor, nb_actions=agent.dim_action, memory=memory, nb_steps_warmup=50,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env,nb_steps=50000,verbose=2,visualize=True)
