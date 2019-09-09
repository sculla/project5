import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPool2D

class Agent(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action
    def model(self):
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
        model.compile(loss='mean_squared_error', optimizer='adam')

        self.model = model


    def act(self, ob, reward, done, vision_on):
        #print("ACT!")

        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        # vision is given as a tensor with size of (64*64, 3) = (4096, 3) <-- rgb
        # and values are in [0, 255]
        if vision_on is False:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel = ob
        else:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, vision = ob

            """ The code below is for checking the vision input. This is very heavy for real-time Control
                So you may need to remove.
            """
            print(vision.shape)
            """
            img = np.ndarray((64,64,3))
            for i in range(3):
                img[:, :, i] = 255 - vision[:, i].reshape((64, 64))

            plt.imshow(img, origin='lower')
            plt.draw()
            plt.pause(0.001)
            """
        return np.tanh(np.random.randn(self.dim_action)) # random action
