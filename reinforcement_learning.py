
import tensorflow as tf


class Deep_model:


    def create_model(self,n_classes=6):

        self.observation_placeholder = tf.placeholder(dtype='float',shape = [None,210,160,3])
        self.rewards_placeholder = tf.placeholder(dtype = 'float' , shape =[None])
        self.action_placeholder = tf.placeholder(dtype='float',shape=[None,n_classes])

        self.avg_reward = tf.reduce_mean(self.rewards_placeholder)
        self.layer1 = tf.layers.conv2d(self.observation_placeholder,filters = 64,kernel_size = [5,5],padding="same",activation=tf.nn.relu)
        self.layer1_flat = tf.layers.flatten(self.layer1)
        self.layer2 = tf.layers.dense(self.layer1_flat,activation=tf.nn.relu,units =30)
        self.prediction = tf.layers.dense(self.layer2,n_classes)

        self.opt = tf.train.AdamOptimizer()
        self.sample_op = tf.multinomial(logits = self.prediction,num_samples=1)


    def setup_train_model(self):
        ##change loss
        self.loss = tf.reduce_sum(self.rewards_placeholder*tf.nn.softmax_cross_entropy_with_logits(logits = self.prediction,labels = self.action_placeholder))
        self.writer = tf.summary.FileWriter('./logs')

        tf.summary.scalar("TOTAL LOSS",self.avg_reward)
        self.merged_summary = tf.summary.merge_all()


        self.train_op = self.opt.minimize(loss=self.loss)


    # def write_graph_and_summary(self,sess,action_data,observation_data,episode):
    #
    #     s = sess.run(self.merged_summary,feed_dict={self.observation_placeholder:observation_data,
    #                                             self.action_placeholder:action_data})
    #     self.writer.add_summary(s,episode)
    #

    def train_model_batches(self,sess,reward_data,action_data,observation_data,episode,batch_size=20):

        m = action_data.shape[0]
        i = 0
        x = i * batch_size

        for i in range(m//batch_size):
            x = i * batch_size
            sess.run(self.train_op,feed_dict={self.observation_placeholder:observation_data[x:x+batch_size,:],
                                               self.action_placeholder:action_data[x:x+batch_size,:],self.rewards_placeholder:reward_data[x:x+batch_size]})
            print("Batch:",i,"trained")
        print(x)
        _,s = sess.run([self.train_op,self.merged_summary],feed_dict={self.observation_placeholder:observation_data[x:,:],
                                          self.action_placeholder:action_data[x:,:],self.rewards_placeholder:reward_data[x:]})
        self.writer.add_summary(s,episode)







#
# if __name__ == '__main__':
#     x = Deep_model()
#     x.create_model()
#
#
#
#
#     print(x.prediction)
#     x.setup_train_model()
