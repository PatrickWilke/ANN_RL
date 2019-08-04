import tensorflow as tf
import numpy as np
import os
from numpy.random import choice


class TrainingNetwork:

    def __init__(self, n_inputs, n_outputs, epsilon, hidden_layers_sizes,
                          discount = 0.99, learning_rate = 0.001, momentum = 0.95):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.epsilon = epsilon

        self.state = tf.placeholder(tf.float32, shape=(None, n_inputs), name='state')
        self.action = tf.placeholder(tf.int64, shape=(None), name='action')
        self.target_value = tf.placeholder(tf.float32, shape=(None), name='target')

        self.reward = tf.placeholder(tf.float32, shape=(None), name='reward')
        self.next_state_weight = tf.placeholder(tf.float32, shape=(None), name='next_state_weight')

        self.hidden_layers = []

        with tf.name_scope('dnn'):

            self.hidden_layers.append(tf.keras.layers.Dense(hidden_layers_sizes[0], name='hidden1',
                                                            activation=tf.nn.relu)(self.state))

            for layer_index in range(1,len(hidden_layers_sizes)):
                self.hidden_layers.append(tf.keras.layers.Dense(hidden_layers_sizes[layer_index],
                                                                name='hidden'+str(layer_index+1),
                                                                activation=tf.nn.relu)(self.hidden_layers[layer_index-1]))
            self.outputs = tf.keras.layers.Dense(n_outputs, name='outputs', activation=tf.keras.activations.linear)(self.hidden_layers[-1])

        self.q_value = tf.reduce_sum(self.outputs * tf.one_hot(self.action, self.n_outputs),axis=1,keepdims=True)
        self.q_max = tf.reduce_max(self.outputs, axis=1)

        with tf.name_scope('loss'):
            self.TD_error = tf.reduce_mean(tf.square(self.target_value - self.q_value))

        self.learning_rate = tf.Variable(learning_rate, name="learning_rate")
        self.momentum = tf.Variable(momentum, name="momentum")

        with tf.name_scope('train'):
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
            self.training_op = self.optimizer.minimize(self.TD_error)

        self.gamma = tf.Variable(discount, name="gamma")
        self.SARSA_target = self.reward +self.next_state_weight*self.gamma*(tf.reduce_sum(self.outputs * tf.one_hot(self.action, self.n_outputs),axis=1))
        self.Q_Learning_target = self.reward + self.next_state_weight*self.gamma*(tf.reduce_max(self.outputs,axis=1))


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
        weights -= np.min(weights)
        if np.sum(weights) == 0:
            return np.random.randint(self.n_outputs)
        probs = weights/np.sum(weights)
        return choice(np.array(range(0,self.n_outputs)),p=probs)

    def CreateSingleReplay(self, game_to_train, is_1v1):

        replay = [[], [], []]
        game_ended = False
        game_to_train.ResetGame()

        if is_1v1:
            current_state = game_to_train.GetState()
            chosen_action = np.random.randint(0, self.n_outputs)
            reward, game_ended = game_to_train.MakeMoveWithReward(chosen_action)

            replay[0].append(current_state)
            replay[1].append(chosen_action)
            replay[2].append(reward)

        while (not game_ended):
            current_state = game_to_train.GetState()
            chosen_action = self.EpsilonGreedyAction(current_state)
            reward, game_ended = game_to_train.MakeMoveWithReward(chosen_action)

            replay[0].append(current_state)
            replay[1].append(chosen_action)
            replay[2].append(reward)

        if is_1v1:
            if 1 < len(replay[2]):

                if (replay[2][-1] == game_to_train.winning_action_reward):
                    replay[2][-2] = game_to_train.loosing_action_reward

        return replay

    def SARSA_Episodic_Single_Game(self, replay, is_1v1):

        if is_1v1:
            shift = 2
        else:
            shift = 1

        target_reward = np.array(replay[2])
        if shift < len(target_reward):
            target_reward[:-shift] += self.gamma.eval()*self.q_value.eval(
                feed_dict={self.state: replay[0][shift:], self.action: replay[1][shift:]}).flatten()

        return target_reward

    def Q_Learning_Episodic_Single_Game(self, replay, is_1v1):

        if is_1v1:
            shift = 2
        else:
            shift = 1

        target_reward = np.array(replay[2])
        if shift < len(target_reward):
            target_reward[:-shift] += self.gamma.eval()* self.q_max.eval(feed_dict={self.state: replay[0][shift:]})

        return target_reward

    def CreateReplay(self, game_to_train, is_1v1, games_per_replay):

        replay = [[], [], [], []]
        count = 0
        while count < games_per_replay:
            count += 1
            game_to_train.ResetGame()
            while (True):

                current_state = game_to_train.GetState()
                chosen_action = self.EpsilonGreedyAction(current_state)
                reward, game_ended = game_to_train.MakeMoveWithReward(chosen_action)

                replay[0].append(current_state)
                replay[1].append(chosen_action)
                replay[2].append(reward)
                if game_ended:
                    replay[3].append(0.0)
                    break
                else:
                    replay[3].append(1.0)

            if is_1v1:
                if 1 < len(replay[2]) and replay[3][-2] == 1.0:
                    replay[3][-2] = 0.0
                    if (replay[2][-1] == game_to_train.winning_action_reward):
                        replay[2][-2] = game_to_train.loosing_action_reward


        replay[0].append(current_state)
        if is_1v1:
            replay[0].append(current_state)

        return replay


    def SARSA_Episodic(self, replay, is_1v1):

        replay[1].append(replay[1][-1])
        if is_1v1:
            shift = 2
            replay[1].append(replay[1][-1])
        else:
            shift = 1

        return self.SARSA_target.eval(feed_dict={self.state: replay[0][shift:], self.action: replay[1][shift:],
                                                          self.reward: replay[2], self.next_state_weight: replay[3]})


    def Q_Learning_Episodic(self, replay, is_1v1):

        if is_1v1:
            shift = 2
        else:
            shift = 1

        return self.Q_Learning_target.eval(feed_dict={self.state: replay[0][shift:],
                                                          self.reward: replay[2], self.next_state_weight: replay[3]})


    def SetUp_File_System(self, path, session):

        root_logdir = "tf_logs"
        log_dir = "{}/{}/".format(root_logdir, path)
        if not os.path.isdir(log_dir):
            file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
            file_writer.close()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        path = "Trainings/" + path

        if os.path.isfile(path + ".index"):
            saver.restore(session, path)
            print("Continue Training")
        else:
            print("New Training")
            init.run()

        return saver, path

    def Training_Episodic_Single_Matches_Reverse(self, game_to_train, store_path, number_of_replays, learning_type, is_1v1 = True,
                                                epsilon_max = 0.15, epsilon_min = 0.02, reverse = True):

        temp_epsilon = self.epsilon

        with tf.Session() as sess:

            saver, path = self.SetUp_File_System(store_path, sess)

            self.epsilon = epsilon_max

            counter = 0
            while counter < number_of_replays:


                replay = self.CreateSingleReplay(game_to_train, is_1v1)
                if not is_1v1:
                    print(np.sum(replay[2]))
                target_reward = learning_type(replay, is_1v1)
                if reverse:
                    for state_it, action_it, target_it in zip(replay[0][::-1], replay[1][::-1], target_reward[::-1]):
                        self.training_op.run(
                            feed_dict={self.state: [state_it], self.action: [action_it], self.target_value: [target_it]})
                else:
                    for state_it, action_it, target_it in zip(replay[0], replay[1], target_reward):
                        self.training_op.run(
                            feed_dict={self.state: [state_it], self.action: [action_it], self.target_value: [target_it]})

                self.epsilon = epsilon_max - (epsilon_max - epsilon_min) * (counter / number_of_replays)
                counter += 1

            saver.save(sess, path)

        self.epsilon = temp_epsilon


    def Training_Episodic_Decorrelated_Batches(self, game_to_train, store_path, number_of_replays, games_per_replay,
                                               batch_size, learning_type,is_1v1 = True, epsilon_max = 0.15, epsilon_min = 0.02):

        temp_epsilon = self.epsilon

        self.lower_bound.assign(game_to_train.prohibited_action_reward)


        with tf.Session() as sess:

            saver, path = self.SetUp_File_System(store_path, sess)

            self.epsilon = epsilon_max

            counter = 0
            while counter < number_of_replays:
                counter += 1

                replay = self.CreateReplay(game_to_train, is_1v1, games_per_replay)
                if not is_1v1:
                    print(replay[2][-1])
                target_reward = learning_type(replay, is_1v1)

                overhead = len(target_reward)%batch_size

                if overhead == 0:
                    index_batch_set = np.random.permutation(len(target_reward)).reshape(-1, batch_size)
                else:
                    index_batch_set = np.random.permutation(len(target_reward))[:-overhead].reshape(-1, batch_size)

                for index_batch in index_batch_set:
                    self.training_op.run(feed_dict={self.state: np.take(replay[0],index_batch,axis=0),
                                               self.action: np.take(replay[1],index_batch),
                                               self.target_value: np.take(target_reward,index_batch)})

                self.epsilon = epsilon_max - (epsilon_max-epsilon_min)*(counter/number_of_replays)

            saver.save(sess, path)

        self.epsilon = temp_epsilon



