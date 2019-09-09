from gym_torcs import TorcsEnv
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPool2D, Reshape

random.seed(42)

run_num = 1


class Agent(object):
    def __init__(self, dim_action) #, random_percent=1, training=True, learning_rate=.96):
        # self.dim_action = dim_action
        # self.episode_memory = []
        # self.lifetime_memory = []
        # self.random_percent = random_percent
        # self.learning_rate = learning_rate
        self.init_model()
        # if not training:
        #     self.model.load_weights()
        # self.training = training
        # y = 0.5 * self.lifetime_memory['reward'] +
        # self.model.fit(x=self.lifetime_memory[['observation', 'new_action']], y=y)

    def init_model(self, dim_action):
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
    def random_action(self):
        return np.tanh(np.random.randn(self.dim_action))

    def act(self, ob, reward, done, vision_on):
        # print("ACT!")

        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        # vision is given as a tensor with size of (64*64, 3) = (4096, 3) <-- rgb
        # and values are in [0, 255]
        if not vision_on:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel = ob
        else:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, vision = ob

        # TODO implement y calculation  = .5 * reward + .1 Decay reward + total_reward
        #  when we have initial observations

        if not self.training:

            if self.random_percent >= random.random():
                action = self.random_action()  # random action
            else:  # hard max for 4 predictions
                pred_in = [[vision, i] for i in range(4)]
                action = np.argmax(self.model.predict(np.array(pred_in)))
            if self.learning_rate > .1:  # decrement the random percent of observations by the learning rate
                self.random_percent *= self.learning_rate
        else:  # Training only
            action = self.random_action()

        if (-1/200) in focus:  # kill episode if off track
            done = True


        return action, done


vision = True
episode_count = 10
max_steps = 100
reward = 0
done = False
step = 0

# Generate a Torcs environment
env = TorcsEnv(vision=vision, throttle=False)

agent = Agent(dim_action=1) #, random_percent=1, training=True, learning_rate=.96)  # steering only

print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    if np.mod(i, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        old_ob = env.reset(relaunch=False)
    else:
        old_ob = env.reset()

    total_reward = 0.
    agent.episode_memory = []
    for j in range(max_steps):
        new_action = agent.act(old_ob, reward, done, vision)
        ob, reward, done, _ = env.step(new_action)
        total_reward += reward
        agent.episode_memory.append({
            "observation": old_ob,
            "new_action": new_action,
            "reward": reward,
            "game_idx": i,
        })
        old_ob = ob
        step += 1
        if done:
            break

    print("TOTAL REWARD @ " + str(i) + " -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("")
    n_steps = len(agent.episode_memory)
    for step_idx, epi_mem in enumerate(agent.episode_memory):
        epi_mem['tot_reward'] = total_reward
        epi_mem['decay_reward'] = step_idx * total_reward / n_steps
    agent.lifetime_memory.extend(agent.episode_memory)

with open(f'output/memory/lifetime_run_{run_num}.pkl') as f:
    pickle.dump(agent.lifetime_memory, f)
agent.save_weights()

env.end()  # This is for shutting down TORCS
print("Finish.")
