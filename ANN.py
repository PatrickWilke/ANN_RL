import tensorflow as tf
import numpy as np
import os
from numpy.random import choice


class TrainingNetwork:

    def __init__(self, n_inputs, n_outputs, epsilon, hidden_layers_sizes):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.epsilon = epsilon

        self.state = tf.placeholder(tf.float32, shape=(None, n_inputs), name='state')
        self.action = tf.placeholder(tf.int64, shape=(None), name='action')
        self.target_value = tf.placeholder(tf.float32, shape=(None), name='target')

        self.hidden_layers = []
        with tf.name_scope('dnn'):

            self.hidden_layers.append(tf.layers.dense(self.state, hidden_layers_sizes[0], name='hidden1', activation=tf.nn.relu))
            for layer_index in range(1,len(hidden_layers_sizes)-1):
                self.hidden_layers.append(tf.layers.dense(self.hidden_layers[layer_index-1], hidden_layers_sizes[layer_index], name='hidden'+str(layer_index+1), activation=tf.nn.relu))
            self.outputs = tf.layers.dense(self.hidden_layers[-1], n_outputs, name='outputs', activation=tf.nn.relu)


    def GetOptimalAction(self, state_given):
        return np.argmax(self.outputs.eval(feed_dict={self.state: [state_given]}))

    def EpsilonGreedyAction(self, state_given):
        if np.random.rand()<self.epsilon:
            return np.random.randint(self.n_outputs)
        else:
            return self.GetOptimalAction(state_given)

    def WeightedAction(self, state_given):
        weights = self.outputs.eval(feed_dict={self.state: [state_given]})[0]
        probs = weights/np.sum(weights)
        return choice(np.array(range(0,9)),p=probs)


    def SARSA_Training(self, game_to_train, store_path, number_of_replays, discount = 0.99, learning_rate = 0.001, momentum = 0.95, epsilon_max = 0.15, epsilon_min = 0.02):

        lower_bound = tf.Variable(game_to_train.prohibited_action_reward, name = "lower_bound")
        q_value = tf.reduce_sum(self.outputs * tf.one_hot(self.action, self.n_outputs), axis=1, keep_dims=True) + lower_bound

        with tf.name_scope('loss'):
            TD_error = tf.reduce_mean(tf.square(self.target_value - q_value))


        with tf.name_scope('train'):
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            training_op = optimizer.minimize(TD_error)


        root_logdir = "tf_logs"
        log_dir = "{}/{}/".format(root_logdir, store_path)
        if not os.path.isdir("./" + log_dir):
            file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
            file_writer.close()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:

            if os.path.isfile(store_path + ".index"):
                saver.restore(sess, store_path)
            else:
                init.run()

            epsilon = epsilon_max

            for k in range(0, number_of_replays):

                replay = [[], [], []]
                game_ended = False
                game_to_train.ResetGame()

                current_state = np.array(game_to_train.GetSate(), copy=True)
                chosen_action = np.random.randint(self.n_outputs)
                reward, game_ended = game_to_train.MakeMoveWithReward(chosen_action)

                replay[0].append(current_state)
                replay[1].append(chosen_action)
                replay[2].append(reward)

                while (not game_ended):

                    current_state = np.array(game_to_train.GetSate(), copy=True)
                    chosen_action = self.EpsilonGreedyAction(current_state)
                    reward, game_ended = game_to_train.MakeMoveWithReward(chosen_action)

                    replay[0].append(current_state)
                    replay[1].append(chosen_action)
                    replay[2].append(reward)


                target_reward = []

                for j in range(0, len(replay[0]) - 2):

                   target_reward.append(discount * q_value.eval(feed_dict={self.state: [replay[0][j + 2]], self.action: [replay[1][j + 2]]}))

                if 1 < len(replay):

                    if (replay[2][-1] == game_to_train.winning_action_reward):
                        target_reward.append(game_to_train.loosing_action_reward)
                    else:
                        target_reward.append(game_to_train.neutral_action_reward)

                target_reward.append(replay[2][-1])
                target_reward = np.array(target_reward)

                for j in reversed(range(0, len(replay[0]))):
                    training_op.run(feed_dict={self.state: [replay[0][j]], self.action: [replay[1][j]], self.target_value: [target_reward[j]]})

                epsilon = epsilon_max - (epsilon_max-epsilon_min)*(k/number_of_replays)

            saver.save(sess, store_path)




