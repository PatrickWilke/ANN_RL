import tensorflow as tf
import numpy as np
import os

epsilon = 0.05

n_inputs = 18
n_hidden1 = 50
n_hidden2 = 50
n_hidden3 = 50
n_outputs = 9


state = tf.placeholder(tf.float32,shape=(None, n_inputs), name='state')
action = tf.placeholder(tf.int64,shape=(None), name='action')
target_value = tf.placeholder(tf.float32,shape=(None), name='target')


with tf.name_scope('dnn'):
    #input_layer = tf.reshape(, [-1, n_inputs])
    hidden1 = tf.layers.dense(state, n_hidden1, name='hidden1', activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1,n_hidden2, name='hidden2', activation=tf.nn.relu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name='hidden3', activation=tf.nn.relu)
    outputs = tf.layers.dense(hidden3,n_outputs, name='outputs', activation=tf.nn.relu)

def GetOptimalAction(state_given):
    return np.argmax(outputs.eval(feed_dict={state: [state_given]}))

def EpsilonGreedyAction(state_given):
    if np.random.rand()<epsilon:
        return np.random.randint(n_outputs)
    else:
        return GetOptimalAction(state_given)


def SARSA_Training(game_to_train, store_path, number_of_replays, discount = 0.99, learning_rate = 0.001, momentum = 0.95, epsilon_max = 0.15, epsilon_min = 0.02):

    lower_bound = tf.Variable(game_to_train.prohibited_action_reward, "lower bound")
    q_value = tf.reduce_sum(outputs * tf.one_hot(action, n_outputs), axis=1, keep_dims=True) + lower_bound

    with tf.name_scope('loss'):
        TD_error = tf.reduce_mean(tf.square(target_value - q_value))


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
            chosen_action = np.random.randint(n_outputs)
            reward, game_ended = game_to_train.MakeMoveWithReward(chosen_action)

            replay[0].append(current_state)
            replay[1].append(chosen_action)
            replay[2].append(reward)

            while (not game_ended):

                current_state = np.array(game_to_train.GetSate(), copy=True)
                chosen_action = EpsilonGreedyAction(current_state)
                reward, game_ended = game_to_train.MakeMoveWithReward(chosen_action)

                replay[0].append(current_state)
                replay[1].append(chosen_action)
                replay[2].append(reward)


            target_reward = []

            for j in range(0, len(replay[0]) - 2):

               target_reward.append(discount * q_value.eval(feed_dict={state: [replay[0][j + 2]], action: [replay[1][j + 2]]}))

            if 1 < len(replay):

                if (replay[2][-1] == game_to_train.winning_action_reward):
                    target_reward.append(game_to_train.loosing_action_reward)
                else:
                    target_reward.append(game_to_train.neutral_action_reward)

            target_reward.append(replay[2][-1])
            target_reward = np.array(target_reward)

            for j in reversed(range(0, len(replay[0]))):
                training_op.run(feed_dict={state: [replay[0][j]], action: [replay[1][j]], target_value: [target_reward[j]]})

            epsilon = epsilon_max - (epsilon_max-epsilon_min)*(k/number_of_replays)

        saver.save(sess, store_path)




