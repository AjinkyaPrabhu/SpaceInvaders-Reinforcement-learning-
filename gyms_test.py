import gym
import numpy as np
import time
import tensorflow as tf
from reinforcement_learning import Deep_model
import os

model = Deep_model()
model.create_model()
model.setup_train_model()

tf.set_random_seed(1)
env = gym.make("SpaceInvaders-v0")
env.reset()

saver = tf.train.Saver()

NUM_EPISODES = 20000



print("POSSIBLE ACIONS ARE:",env.action_space)

"""
   0 : "NOOP",
   1 : "FIRE",
   2 : "UP",
   3 : "RIGHT",
   4 : "LEFT",
   5 : "DOWN",


"""

def countDown(x):
    for i in range(x):
        time.sleep(1)
        print(x-i)




countDown(1)




def to_one_hot(index):
    x = np.zeros(6)
    x[index] = 1
    return x


def create_new_reward(reward):
    return reward-1000


def finish_episode():
    pass




with tf.Session() as sess:

    print("Trying to restore saved model")
    if  os.path.isdir("./saved_models/"):
        saver.restore(sess,'./saved_models/model.ckpt')
        print("Model Restored")
    else:
        print("No Saved Model found....Starting Fresh")

    sess.run(tf.global_variables_initializer())
    for episode in range(NUM_EPISODES):
        l1 = 3
        observation  = env.reset()
        reward_data = []
        observation_data = []
        action_data = []
        time_steps = 0
        while time_steps<10000:
            time_steps+=1
            env.render()
            print(time_steps)
            observation = observation.reshape((1,210,160,3))

            action_sampled = sess.run(model.sample_op,feed_dict={model.observation_placeholder:observation})
            action = action_sampled[0,0]
            #dropout type


            observation,reward,done,info = env.step(action)





            # l2 = info['ale.lives']
            # if l2<l1:
            #     l1 = l2
            #     reward = create_new_reward(reward)



            reward_data.append(reward)
            action_data.append(to_one_hot(action_sampled[0,0]))
            observation_data.append(observation)





            if done:

                action_data = np.array(action_data)
                observation_data = np.array(observation_data)
                reward_data = np.array(reward_data) #sum(reward_data)* np.ones((action_data.shape[0])) #
                final_rewards = []
                R = 0
                for r  in reward_data[::-1]:
                    R = r + 0.7 * R
                    final_rewards.insert(0,R)

                final_rewards = np.array(final_rewards)

                final_rewards = (final_rewards-final_rewards.mean())/(final_rewards.std() + 1e-4)
                print(final_rewards)
                print(final_rewards.shape)
                print(action_data.shape)
                print(observation_data.shape)
                model.train_model_batches(sess,final_rewards,action_data,observation_data,episode=episode)
                # model.write_graph_and_summary(sess,action_data,observation_data,episode)
                print("Episode done after {} timesteps",time_steps)
                print("Saving model.....")
                saver.save(sess,'./saved_models/model.ckpt')

                break
