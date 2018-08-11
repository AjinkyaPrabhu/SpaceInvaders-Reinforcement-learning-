
import tensorflow as tf
class Deep_model:


    def create_model(self,n_classes=6):



        self.observation_placeholder = tf.placeholder(dtype='float',shape = [None,210,160,3])
        self.rewards_placeholder = tf.placeholder(dtype = float , shape =[None])
        self.action_placeholder = tf.placeholder(dtype='float',shape=[None,n_classes])


        self.layer1 = tf.layers.conv2d(self.observation_placeholder,filters = 64,kernel_size = [5,5],padding="same",activation=tf.nn.relu)
        self.layer1_flat = tf.layers.flatten(self.layer1)
        self.prediction = tf.layers.dense(self.layer1_flat,n_classes)

        self.opt = tf.train.AdamOptimizer()
        self.sample_op = tf.multinomial(logits = self.prediction,num_samples=1)


    def setup_train_model(self):
        ##change loss
        self.loss = tf.reduce_sum(self.rewards_placeholder*tf.nn.softmax_cross_entropy_with_logits(logits = self.prediction,labels = self.action_placeholder))
        self.train_op = self.opt.minimize(loss=self.loss)


    # def write_graph_and_summary(self,sess,action_data,observation_data):
    #     writer = tf.summary.FileWriter('./logs')
    #     writer.add_graph(graph=sess.graph)
    #     tf.summay.scalar("TOTAL LOSS",self.loss)
    #     merged_summary = tf.summary.merge_all()
    #
    #     sess.run(merged_summary,feed_dict={self.observation_placeholder=observation_data,self.action_placeholder:action_data})
    #






if __name__ == '__main__':
    x = Deep_model()
    x.create_model()

    print(x.prediction)
    x.setup_train_model()
