import gym
import numpy as np
import time
import tensorflow as tf
from reinforcement_learning import Deep_model


model = Deep_model()
model.create_model()
model.setup_train_model()


env = gym.make("SpaceInvaders-v0")
env.reset()

NUM_EPISODES = 20000



print("POSSIBLE ACIONS ARE:",env.action_space)

"""
   0 : "NOOP",
   1 : "FIRE",
   2 : "UP",
   3 : "RIGHT",
   4 : "LEFT",
   5 : "DOWN",
   6 : "UPRIGHT",
   7 : "UPLEFT",
   8 : "DOWNRIGHT",
   9 : "DOWNLEFT",
   10 : "UPFIRE",
   11 : "RIGHTFIRE",
   12 : "LEFTFIRE",
   13 : "DOWNFIRE",
   14 : "UPRIGHTFIRE",
   15 : "UPLEFTFIRE",
   16 : "DOWNRIGHTFIRE",
   17 : "DOWNLEFTFIRE",

"""

def countDown(x):
    for i in range(x):
        time.sleep(1)
        print(x-i)




countDown(4)

reward_data = []
observation_data = []
action_data = []


def to_one_hot(index):
    x = np.zeros(6)
    x[index] = 1
    return x

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode in range(NUM_EPISODES):
        observation  = env.reset()

        time_steps = 0
        while True:
            time_steps+=1
            env.render()

            observation = observation.reshape((1,210,160,3))
            action_sampled = sess.run(model.sample_op,feed_dict={model.observation_placeholder:observation})


            observation,reward,done,info = env.step(action_sampled[0,0])

            reward_data.append(reward)
            action_data.append(to_one_hot(action_sampled[0,0]))
            observation_data.append(observation)

            print(reward)
            if done:
                print(reward_data)
                print(action_data)
                print(observation_data)
                sess.run(model.train_op,feed_dict={model.observation_placeholder:np.array(observation_data),
                model.action_placeholder:np.array(action_data),model.rewards_placeholder:np.array(reward_data)})
                print("Episode done after {} timesteps",time_steps)
                break
