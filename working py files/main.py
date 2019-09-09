import gym
import pandas as pd
import numpy as np
import time
import sys

import numpy as np
import gym

import tensorflow as tf
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, RNN
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from keras.applications.vgg16 import VGG16
from keras import optimizers

#sparse_categorical_crossentropy

def build_model():
    model = Sequential()
    model.add(Dense(units=10, input_dim=2, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    # model.add(Dense(units=10, activation='relu'))
    # model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    adam = Adam() #lr=.01, decay=1e-6)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=adam)
    # model = Sequential()
    # model.add(Dense(40,input_dim=2, activation='relu'))
    # # model.add(Dense(30, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer=adam)
    model.build()
    print(model.summary())
    return model


    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # # even the metrics!
    # memory = SequentialMemory(limit=50000, window_length=1)
    # policy = BoltzmannQPolicy()
    # dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
    #                target_model_update=1e-2, policy=policy)
    # dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    #
    # # Okay, now it's time to learn something! We visualize the training here for show, but this
    # # slows down training quite a lot. You can always safely abort the training prematurely using
    # # Ctrl + C.
    # dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
    #
    # # After training is done, we save the final weights.
    # dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    #
    # # Finally, evaluate our algorithm for 5 episodes.
    # dqn.test(env, nb_episodes=5, visualize=True)


def train_random_action(n_iter=int(5e5)):
    life_memory = []
    env = gym.make('FrozenLake-v0')  # build a fresh environment
    tot_reward = 0
    for _game_idx in range(n_iter):
        old_observation = env.reset()  # this starts a new "episode" and returns the initial observation
        done = False

        episode_mem = []
        while not done:
            new_action = env.action_space.sample()
            observation, reward, done, info = env.step(new_action)
            tot_reward += reward
            episode_mem.append(dict(zip(['observation', 'new_action', 'reward', 'game_idx'],
                                        (old_observation, new_action, reward, _game_idx))))
            old_observation = observation
        n_steps = len(episode_mem)
        for step_idx, epi_mem in enumerate(episode_mem):
            epi_mem['tot_reward'] = tot_reward
            epi_mem['decay_reward'] = step_idx * tot_reward / n_steps
        life_memory.extend(episode_mem)

    g_df = pd.DataFrame(life_memory)
    g_df.to_pickle('data/initial_games.pkl')


if __name__ == '__main__':
    GPU = True
    CPU = False

    num_cores = 4
    if GPU:
        num_GPU = 1
        num_CPU = 1
    if CPU:
        num_CPU = 1
        num_GPU = 0
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU}
                            )
    session = tf.Session(config=config)
    K.set_session(session)

    env = gym.make('FrozenLake-v0')

    #
    # memory = SequentialMemory(limit=50000, window_length=1)
    # policy = BoltzmannQPolicy()
    # dqn = DQNAgent(model=build_model(), nb_actions=4, memory=memory, nb_steps_warmup=10,
    #                target_model_update=1e-2, policy=policy)
    # dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    # dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
    #

    toolbar_width = 100
    # setup toolbar

    trained = False
    from sklearn.ensemble import ExtraTreesRegressor

    train_random_action(int(1e4))
    random_per = 0.5
    for n in range(50):
        if trained:
            df = pd.read_pickle('data/intermediate_games1.pkl')
            if n == 1:
                old_mean = np.mean(df.groupby('game_idx').reward.sum())
                print(f'Old mean is : {old_mean}')
            y = 0.5 * df.reward + 0.1 * df.decay_reward + df.tot_reward
            y = np.array(y)
            x = df[["observation", "new_action"]]
            print('training')
            # model = ExtraTreesRegressor(n_estimators=50)
            model = build_model()
            model.fit(x, y, verbose=1)
            print('trained')
            n_passed = 0

        elif n == 0:
            df = pd.read_pickle('data/initial_games.pkl')
            old_mean = np.mean(df.groupby("game_idx").reward.sum())
            print(old_mean)
            y = 0.5 * df.reward + 0.1 * df.decay_reward + df.tot_reward
            y = np.array(y)
            x = df[["observation", "new_action"]]
            print('training')
            # model = ExtraTreesRegressor(n_estimators=50)

            model = build_model()
            model.fit(x, y, verbose=1)
            print('trained')
            n_passed = 0

        else:
            print('passed')
            n_passed += 1

        num_episodes = int(1e4)

        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width + 1))

        life_memory = []
        for i in range(num_episodes):

            if i % 100 == 0:
                sys.stdout.write("-")
                sys.stdout.flush()
            # start a new episode and record all the memories
            old_observation = env.reset()
            done = False
            tot_reward = 0
            ep_memory = []
            while not done:

                if np.random.rand() < random_per:
                    new_action = env.action_space.sample()
                else:
                    pred_in = [[old_observation, i] for i in range(4)]
                    new_action = np.argmax(model.predict(np.array(pred_in)))
                observation, reward, done, info = env.step(new_action)
                tot_reward += reward

                ep_memory.append({
                    "observation": old_observation,
                    "new_action": new_action,
                    "reward": reward,
                    "game_idx": i,
                })
                old_observation = observation
            n_steps = len(ep_memory)
            # incorporate total reward
            for step_idx, epi_mem in enumerate(ep_memory):
                epi_mem['tot_reward'] = tot_reward
                epi_mem['decay_reward'] = step_idx * tot_reward / n_steps

            life_memory.extend(ep_memory)

        memory_df2 = pd.DataFrame(life_memory)

        # rf.fit(memory_df[["observation", "action"]], memory_df["comb_reward"])

        # score
        # much better!
        sys.stdout.write("]\n")  # this ends the progress bar
        new_mean = np.mean(memory_df2.groupby('game_idx').reward.sum())
        print(f'\nnew mean is: {new_mean}, old mean is: {old_mean}, random percent: {random_per}')
        memory_df2.to_pickle('data/keras_prev_intermediate_games1.pkl')

        if new_mean > old_mean:
            old_mean = new_mean
            memory_df2.to_pickle('data/keras_intermediate_games1.pkl')
            trained = True
            random_per += .05
            if random_per > 1:
                random_per = 1

            print(f'random percent updated to: {random_per}')
        else:
            trained = False
            random_per -= .05
            if random_per < 0:
                random_per = 0

            print(f'random percent updated to: {random_per}')
        if n_passed > 3:
            random_per = 1
            print('reset random!!')


