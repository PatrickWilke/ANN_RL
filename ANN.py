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
            for layer_index in range(1,len(hidden_layers_sizes)):
                self.hidden_layers.append(tf.layers.dense(self.hidden_layers[layer_index-1], hidden_layers_sizes[layer_index], name='hidden'+str(layer_index+1), activation=tf.nn.relu))
            self.outputs = tf.layers.dense(self.hidden_layers[-1], n_outputs, name='outputs', activation=tf.nn.relu)

        self.lower_bound = tf.Variable(0.0, name="lower_bound")

        self.q_value = tf.reduce_sum(self.outputs * tf.one_hot(self.action, self.n_outputs),axis=1,keepdims=True) + self.lower_bound

        self.q_max = tf.reduce_max(self.outputs,axis=1) + self.lower_bound

    def GetOptimalAction(self, state_given):
        list_of_q_values = self.outputs.eval(feed_dict={self.state: [state_given]})[0]
        all_candidates = np.argwhere(list_of_q_values == np.amax(list_of_q_values))
        return np.random.choice(all_candidates.flatten())

    def EpsilonGreedyAction(self, state_given):
        if np.random.rand()<self.epsilon:
            return np.random.randint(self.n_outputs)
        else:
            return self.GetOptimalAction(state_given)

    def WeightedAction(self, state_given):
        weights = self.outputs.eval(feed_dict={self.state: [state_given]})[0]
        if np.sum(weights) == 0:
            return np.random.randint(self.n_outputs)
        probs = weights/np.sum(weights)
        return choice(np.array(range(0,self.n_outputs)),p=probs)

    def CreateSingleReplay(self, game_to_train):

        replay = [[], [], []]
        game_ended = False
        game_to_train.ResetGame()

        current_state = game_to_train.GetSate()
        chosen_action = np.random.randint(0,self.n_outputs)
        reward, game_ended = game_to_train.MakeMoveWithReward(chosen_action)

        replay[0].append(current_state)
        replay[1].append(chosen_action)
        replay[2].append(reward)

        while (not game_ended):
            current_state = game_to_train.GetSate()
            chosen_action = self.EpsilonGreedyAction(current_state)
            reward, game_ended = game_to_train.MakeMoveWithReward(chosen_action)

            replay[0].append(current_state)
            replay[1].append(chosen_action)
            replay[2].append(reward)

        if 1 < len(replay[2]):

            if (replay[2][-1] == game_to_train.winning_action_reward):
                replay[2][-2] = game_to_train.loosing_action_reward

        return replay

    def SARSA_1v1_Episodic(self, replay, discount):


        target_reward = np.array(replay[2])
        if 2 < len(target_reward):
            target_reward[:-2] += discount * self.q_value.eval(feed_dict={self.state: replay[0][2:], self.action: replay[1][2:]}).flatten()

        return target_reward

    def Q_Learning_1v1_Episodic(self, replay, discount):

        target_reward = np.array(replay[2])
        if 2 < len(target_reward):
            target_reward[:-2] += discount * self.q_max.eval(feed_dict={self.state: replay[0][2:]})

        return target_reward



    def Training_1v1_Episodic(self, game_to_train, store_path, number_of_replays, discount = 0.99, learning_rate = 0.001, momentum = 0.95, epsilon_max = 0.15, epsilon_min = 0.02):

        temp_epsilon = self.epsilon

        self.lower_bound.assign(game_to_train.prohibited_action_reward)

        with tf.name_scope('loss'):
            TD_error = tf.reduce_mean(tf.square(self.target_value - self.q_value))


        with tf.name_scope('train'):
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            training_op = optimizer.minimize(TD_error)

        root_logdir = "tf_logs"
        log_dir = "{}/{}/".format(root_logdir, store_path)
        if not os.path.isdir(log_dir):
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

                replay = self.CreateSingleReplay(game_to_train)

                target_reward = self.Q_Learning_1v1_Episodic(replay, discount)

                for j in reversed(range(0, len(target_reward))):
                    training_op.run(feed_dict={self.state: [replay[0][j]], self.action: [replay[1][j]], self.target_value: [target_reward[j]]})

                epsilon = epsilon_max - (epsilon_max-epsilon_min)*(k/number_of_replays)

            saver.save(sess, store_path)

        self.epsilon = temp_epsilon



